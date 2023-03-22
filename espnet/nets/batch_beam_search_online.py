"""Parallel beam search module for online simulation."""

import logging
import time
from typing import Any  # noqa: H301
from typing import Dict  # noqa: H301
from typing import List  # noqa: H301
from typing import Tuple  # noqa: H301

import copy
import torch

from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: H301
from espnet.nets.batch_beam_search import BatchHypothesis  # noqa: H301
from espnet.nets.beam_search import Hypothesis
from espnet.nets.e2e_asr_common import end_detect


class BatchBeamSearchOnline(BatchBeamSearch):
    """Online beam search implementation.

    This simulates streaming decoding.
    It requires encoded features of entire utterance and
    extracts block by block from it as it shoud be done
    in streaming processing.
    This is based on Tsunoo et al, "STREAMING TRANSFORMER ASR
    WITH BLOCKWISE SYNCHRONOUS BEAM SEARCH"
    (https://arxiv.org/abs/2006.14941).
    """

    def __init__(
        self,
        *args,
        block_size=40,
        hop_size=16,
        look_ahead=16,
        disable_repetition_detection=False,
        encoded_feat_length_limit=0,
        decoder_text_length_limit=0,
        incremental_decode=False,
        incremental_strategy=None,
        ctc_wait=None,
        length_penalty=None,
        recompute_decoder_states=False,
        **kwargs,
    ):
        """Initialize beam search."""
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.hop_size = hop_size
        self.look_ahead = look_ahead
        self.disable_repetition_detection = disable_repetition_detection
        self.encoded_feat_length_limit = encoded_feat_length_limit
        self.decoder_text_length_limit = decoder_text_length_limit
        self.incremental_decode = incremental_decode
        self.length_penalty = length_penalty
        
        self.hold_n = -1
        self.wait_k = -1
        self.wait_k_interval = -1
        self.local_agreement = -1
        self.ctc_wait = ctc_wait if ctc_wait else -1
        self.ctc_greedy_len = 0

        self.recompute_decoder_states = recompute_decoder_states

        if incremental_strategy:
            if incremental_strategy.startswith('hold-'):
                self.hold_n = int(incremental_strategy[5:])
            if incremental_strategy.startswith('wait-'):
                waitk_args = incremental_strategy.split('-')[1:]
                self.wait_k_interval = int(waitk_args[0])
                self.wait_k = int(waitk_args[1])
            elif incremental_strategy.startswith('local-agreement-'):
                self.local_agreement = int(incremental_strategy[16:])
            else:
                logging.error('Unsupported incremental strategy. Supported: hold-N and local-agreement-K.')


        self.reset()

    def reset(self):
        """Reset parameters."""
        self.encbuffer = None
        self.running_hyps = None
        self.prev_hyps = None
        self.stable = torch.tensor([])
        self.ended_hyps = []
        self.processed_block = 0
        self.process_idx = 0
        self.prev_output = None
        self.seen_unreliable_hyps = dict()

    def score_full(
        self, hyp: BatchHypothesis, x: torch.Tensor, pre_x: torch.Tensor = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (Hypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        scores = dict()
        states = dict()
        for k, d in self.full_scorers.items():
            if (
                self.decoder_text_length_limit > 0
                and len(hyp.yseq) > 0
                and len(hyp.yseq[0]) > self.decoder_text_length_limit
            ):
                temp_yseq = hyp.yseq.narrow(
                    1, -self.decoder_text_length_limit, self.decoder_text_length_limit
                ).clone()
                temp_yseq[:, 0] = self.sos
                self.running_hyps.states["decoder"] = [
                    None for _ in self.running_hyps.states["decoder"]
                ]
                scores[k], states[k] = d.batch_score(temp_yseq, hyp.states[k], x)
            else:
                scores[k], states[k] = d.batch_score(hyp.yseq, hyp.states[k], x)
        return scores, states

    def forward(
        self,
        x: torch.Tensor,
        maxlenratio: float = 0.0,
        minlenratio: float = 0.0,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        if x.shape[0] == 0:
            return []
        if self.encbuffer is None:
            self.encbuffer = x
        else:
            self.encbuffer = torch.cat([self.encbuffer, x], axis=0)

        x = self.encbuffer

        # set length bounds
        if maxlenratio == 0:
            maxlen = x.shape[0]
        else:
            maxlen = max(1, int(maxlenratio * x.size(0)))

        ret = None
        if x.shape[0] > 0:
            block_is_final = is_final
            h = x
            logging.info(f'processing {h.shape[0]} out of {x.shape[0]}')

            logging.debug("Start processing block: %d", self.processed_block)
            logging.debug(
                "  Feature length: {}, current position: {}".format(
                    h.shape[0], self.process_idx
                )
            )
            if (
                self.encoded_feat_length_limit > 0
                and h.shape[0] > self.encoded_feat_length_limit
            ):
                h = h.narrow(
                    0,
                    h.shape[0] - self.encoded_feat_length_limit,
                    self.encoded_feat_length_limit,
                )

            if self.running_hyps is None:
                self.running_hyps = self.init_hyp(h)
            ret = self.process_one_block(h, block_is_final, maxlen, maxlenratio)
            logging.debug("Finished processing block: %d", self.processed_block)
            self.processed_block += 1
            if block_is_final:
                return ret
        if ret is None:
            if self.prev_output is None:
                return []
            else:
                return self.prev_output
        else:
            self.prev_output = ret
            # N-best results
            return ret

    def add_seen_unreliable_hypo(self, seq):
        seq = seq.cpu().numpy()
        unreliable = self.seen_unreliable_hyps
        for t in seq:
            if t not in unreliable:
                unreliable[t] = {}
            unreliable = unreliable[t]

    def is_in_seen_unreliable_hypo(self, seq):
        seq = seq.cpu().numpy()
        unreliable = self.seen_unreliable_hyps
        for t in seq:
            if t in unreliable:
                unreliable = unreliable[t]
            else:
                return False
        return True

    def _contains_punct(self, token_id):
        return any(
            p in self.token_list[token_id] 
            for p in list('.,?!():&')
        )

    def process_one_block(self, h, is_final, maxlen, maxlenratio):
        """Recognize one block."""
        # extend states for ctc
        if self.recompute_decoder_states:
            self.running_hyps = self.init_hyp(h)

        self.extend(h, self.running_hyps)

        if self.recompute_decoder_states:
            for token in self.stable[1:]:
                self.running_hyps = self.search(
                    self.running_hyps, 
                    h, 
                    force=token.unsqueeze(-1)
                )
        
        if not is_final:
            s = time.time()
            prev_hypo = copy.deepcopy(self.running_hyps)
            logging.info(f'copy time: {time.time() - s}')
            
            if self.ctc_wait > -1:
                maxlen = self.ctc_greedy_len - self.ctc_wait + 1
            
            if self.wait_k > -1:
                logging.info(h.shape)
                maxlen = h.shape[0] // self.wait_k_interval - self.wait_k
        
            if self.local_agreement > 0 and self.processed_block % self.local_agreement > 0:
                return None

        finished_hyps = []
        max_unreliable_score_score = -float('inf') #anything <= is also considered unreliable
        while self.process_idx < maxlen:
            logging.debug("position " + str(self.process_idx))
            best = self.search(self.running_hyps, h)

            # We reduce the beam search by the number of finished hypotheses
            if len(best) > self.beam_size - len(finished_hyps):
                best = self._batch_select(best, list(range(self.beam_size - len(finished_hyps))))
                assert len(best) > 0

            if self.process_idx == maxlen - 1:
                best.yseq[:, best.length - 1] = self.eos

            n_batch = best.yseq.shape[0]
            finished = best.yseq[torch.arange(n_batch), best.length - 1] == self.eos

            for i in range(n_batch):
                # Always remove beams with EOS 

                if (
                        finished[i]
                        or (best.finished[i] and not is_final) 
                        or (
                            not self.disable_repetition_detection
                            and (
                                (
                                    len(self.token_list[best.yseq[i, -1].item()].replace('▁', '')) > 0 
                                    and best.yseq[i, -1] in best.yseq[i, :-1]
                                )   # repetition
                                or best.score[i] <= max_unreliable_score_score
                            )
                            # We allow repetitions if generated again with more context
                            and not self.is_in_seen_unreliable_hypo(best.yseq[i]) 
                            and not is_final
                        )
                ):
                    hyp = self._batch_select(best, [i,])
                    finished[i] = True
                    finished_hyps.append(hyp)
                    max_unreliable_score_score = max(max_unreliable_score_score, hyp.score[0])
                    self.add_seen_unreliable_hypo(hyp.yseq[0])

            # remove finished/unreliable beams 
            best = self._batch_select(best, torch.nonzero(finished == False).view(-1))

            if len(best) == 0:
                logging.info("All beams finished.")
                break
            else:
                self.running_hyps = self.post_process(
                    #self.process_idx, maxlen, maxlenratio, best, self.ended_hyps
                    self.process_idx, maxlen, maxlenratio, best, []
                )

            # increment number
            self.process_idx += 1

        if is_final:
            finished_hyps = [self._select(h, 0) for h in finished_hyps]
            return self.assemble_hyps(finished_hyps)
        elif len(finished_hyps) > 0:
            # Sort by length-normalized score
            rets = sorted(finished_hyps, key=lambda x: self._norm_score(x.score, x.length[0]), reverse=True)
            best : BatchHypothesis = rets[0]
            # Always keep SOS token 
            stable_length = max(1, best.length[0] - 1)
            if self.hold_n > -1:
                stable_length = max(1, stable_length - self.hold_n)
            elif self.local_agreement > 0:
                stable_length = self._compute_local_agreement(best)
            stable_length = max(
                stable_length,
                prev_hypo.length[0]
            )
            self.prev_hyps = best.yseq[0]
            best = self._batch_select_stable_prefix(best, stable_length)
            self.stable = best.yseq[0]
            self.process_idx = stable_length - 1
            
            if not self.recompute_decoder_states:
                s = time.time()
                self.running_hyps = self._update_ctc_state(prev_hypo, best)
                logging.info(f'update time {time.time() - s}')
            assert self.running_hyps.yseq.shape[0] == 1
            logging.info(
                ''.join(
                    [self.token_list[t.item()] for t in self.running_hyps.yseq[0]]
                )
            )
            return [self._select(best, 0),]
        return None

    def _norm_score(self, score, l):
        if self.length_penalty is not None:
            return score + self.length_penalty * (l - 1)
        elif l > 1:
            return score / (l - 1)
        else:
            return score

    def _update_ctc_state(self, prev_hypo : BatchHypothesis, current_hypo : BatchHypothesis) -> BatchHypothesis:
        prev_hypo_len = prev_hypo.length[0]
        current_hypo_len = current_hypo.length[0]
        for scorer_name, part_scorer in self.part_scorers.items():
            if 'ctc' in scorer_name:
                new_state = prev_hypo.states[scorer_name]
                for token_idx in torch.arange(prev_hypo_len, current_hypo_len):
                    curr_y = current_hypo.yseq[0,:token_idx].view((1,-1))
                    new_token = current_hypo.yseq[0,token_idx].view((1,1,-1))
                    _, new_state = part_scorer.batch_score_partial(
                        curr_y,
                        new_token,
                        new_state,
                        None, # x
                    )
                    new_state = part_scorer.select_state(
                        new_state,
                        0,
                        new_id=current_hypo.yseq[0,token_idx].item()
                    )
                    new_state = [new_state,]
                current_hypo.states[scorer_name] = new_state
        return current_hypo


    def _compute_local_agreement(self, best: BatchHypothesis):
        if self.prev_hyps is None:
            return 1
        prev = self.prev_hyps
        best = best.yseq[0,:-1]
        for idx, (pt, bt) in enumerate(zip(prev, best)):
            if pt != bt:
                break
        return idx

    def assemble_hyps(self, ended_hyps):
        """Assemble the hypotheses."""
        nbest_hyps = sorted(ended_hyps, key=lambda x: self._norm_score(x.score, len(x.yseq)), reverse=True)
        # check the number of hypotheses reaching to eos
        if len(nbest_hyps) == 0:
            logging.warning(
                "there is no N-best results, perform recognition "
                "again with smaller minlenratio."
            )
            return []

        # report the best result
        best = nbest_hyps[0]
        for k, v in best.scores.items():
            logging.info(
                f"{v[-1].item():6.2f} * {self.weights[k]:3} = {v[-1].item() * self.weights[k]:6.2f} for {k}"
            )
        logging.info(f"total log probability: {best.score:.2f}")
        logging.info(f"normalized log probability: {best.score / len(best.yseq):.2f}")
        logging.info(f"total number of ended hypotheses: {len(nbest_hyps)}")
        if self.token_list is not None:
            logging.info(
                "best hypo: "
                + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                + "\n"
            )
        return nbest_hyps

    def extend(self, x: torch.Tensor, hyps: Hypothesis) -> List[Hypothesis]:
        """Extend probabilities and states with more encoded chunks.

        Args:
            x (torch.Tensor): The extended encoder output feature
            hyps (Hypothesis): Current list of hypothesis

        Returns:
            Hypothesis: The extended hypothesis

        """
        for k, d in self.scorers.items():
            if hasattr(d, "extend_prob"):
                d.extend_prob(x)
            if hasattr(d, "extend_state"):
                hyps.states[k] = d.extend_state(hyps.states[k])
            if hasattr(d, "get_ctc_greedy_len"):
                self.ctc_greedy_len = d.get_ctc_greedy_len()
