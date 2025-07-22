import gc

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from llm_attacks import AttackPrompt, TargetsPrompt, MultiPromptAttack, PromptManager
from llm_attacks import get_embedding_matrix, get_embeddings, process_targets


def token_gradients(model, tokenizer, target_str, input_ids, input_slice, target_slice, loss_slice):

    """
    Computes gradients of the loss with respect to the coordinates.
    
    Parameters
    ----------
    model : Transformer Model
        The transformer model to be used.
    input_ids : torch.Tensor
        The input sequence in the form of token ids.
    input_slice : slice
        The slice of the input sequence for which gradients need to be computed.
    target_slice : slice
        The slice of the input sequence to be used as targets.
    loss_slice : slice
        The slice of the logits to be used for computing the loss.

    Returns
    -------
    torch.Tensor
        The gradients of each token in the input_slice with respect to the loss.
    """
    assert target_str.startswith('Sure, here '), "Target must be start with 'Sure, here'"
    # Prepare new targets
    t = tokenizer.decode(input_ids[target_slice])
    targets = process_targets(target_str)
    
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype
    )
    one_hot.scatter_(
        1, 
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype)
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)
    
    losses = []
    # Prepare input_ids and slices
    embeds = get_embeddings(model, input_ids[:target_slice.start].unsqueeze(0)).detach()
    body_embeds = torch.cat(
        [
            embeds[:,:input_slice.start,:], 
            input_embeds, 
            embeds[:,input_slice.stop:,:]
        ], 
        dim=1)
    outputs = model(inputs_embeds=body_embeds, use_cache=True)
    logits_body = outputs.logits
    past_key_values = outputs.past_key_values
    for t in targets:
        target_ids = torch.tensor(tokenizer.encode(t, add_special_tokens=False)).to(input_ids.device).unsqueeze(0)
        target_slice = slice(target_slice.start, target_slice.start+target_ids.shape[1])
        loss_slice = slice(loss_slice.start, target_slice.start+target_ids.shape[1]-1)
        
        # Computing logits of different targets.
        position_ids_target = torch.arange(target_slice.start, target_slice.stop, device=input_ids.device).unsqueeze(0)
        logits_target = model(
                input_ids=target_ids,
                past_key_values = past_key_values,
                position_ids = position_ids_target,
                use_cache = True,
            ).logits
        logits = torch.cat([logits_body, logits_target], dim=1)
        loss = nn.CrossEntropyLoss()(logits[0,loss_slice,:], target_ids[0])
        losses.append(loss)
        
    del embeds, body_embeds, logits_body, logits_target, past_key_values, logits, target_ids; gc.collect()
    torch.cuda.empty_cache()

    confidences = [-loss for loss in losses] 
    softmax_weights = torch.softmax(torch.tensor(confidences), dim=0)
    total_loss = sum(w * loss for w, loss in zip(softmax_weights, losses))
    total_loss.backward()
    
    return one_hot.grad.clone(), losses.index(min(losses))
    # return one_hot.grad.clone()

class GCGAttackPrompt(AttackPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def grad(self, model):
        return token_gradients(
            model, 
            self.tokenizer, 
            self.target_str,
            self.input_ids.to(model.device), 
            self._control_slice, 
            self._target_slice, 
            self._loss_slice
        )

class GCGTargetsPrompt(TargetsPrompt):

    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def grad(self, model):
        return self.prompts_list[0].grad(model) 

class GCGPromptManager(PromptManager):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def sample_control(self, grad, batch_size, topk=256, temp=1, allow_non_ascii=True):

        if not allow_non_ascii:
            grad[:, self._nonascii_toks.to(grad.device)] = np.infty
        top_indices = (-grad).topk(topk, dim=1).indices
        control_toks = self.control_toks.to(grad.device)
        original_control_toks = control_toks.repeat(batch_size, 1)
        new_token_pos = torch.arange(
            0, 
            len(control_toks), 
            len(control_toks) / batch_size,
            device=grad.device
        ).type(torch.int64)
        new_token_val = torch.gather(
            top_indices[new_token_pos], 1, 
            torch.randint(0, topk, (batch_size, 1),
            device=grad.device)
        )
        new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
        return new_control_toks


class GCGMultiPromptAttack(MultiPromptAttack):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def step(self, 
             batch_size=1024, 
             topk=256, 
             temp=1, 
             allow_non_ascii=True, 
             target_weight=1, 
             control_weight=0.1, 
             verbose=False, 
             opt_only=False,
             filter_cand=True):

        
        # GCG currently does not support optimization_only mode, 
        # so opt_only does not change the inner loop.
        opt_only = False

        main_device = self.models[0].device
        control_cands = []

        # for j, worker in enumerate(self.workers):
        #     worker(self.prompts[j], "grad", worker.model)

        # Aggregate gradients
        grad = None
        for j, worker in enumerate(self.workers):
            # new_grad = worker.results.get().to(main_device)
            new_grad, target_idx = self.prompts[j].grad(worker.model)
            new_grad.to(main_device)
            new_grad = new_grad / new_grad.norm(dim=-1, keepdim=True)
            if grad is None:
                grad = torch.zeros_like(new_grad)
            if grad.shape != new_grad.shape:
                with torch.no_grad():
                    control_cand = self.prompts[j-1].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
                    control_cands.append(self.get_filtered_cands(j-1, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
                grad = new_grad
            else:
                grad += new_grad

        with torch.no_grad():
            control_cand = self.prompts[j].sample_control(grad, batch_size, topk, temp, allow_non_ascii)
            control_cands.append(self.get_filtered_cands(j, control_cand, filter_cand=filter_cand, curr_control=self.control_str))
        del grad, control_cand ; gc.collect()
        
        # Search
        loss = torch.zeros(len(control_cands) * batch_size).to(main_device)
        with torch.no_grad():
            for j, cand in enumerate(control_cands):
                # Looping through the prompts at this level is less elegant, but
                # we can manage VRAM better this way
                progress = tqdm(range(len(self.prompts[0])), total=len(self.prompts[0])) if verbose else enumerate(self.prompts[0])
                for i in progress:
                    # for k, worker in enumerate(self.workers):
                    #     worker(self.prompts[k][i][target_idx], "logits", worker.model, cand, return_ids=True)
                    # logits, ids = zip(*[worker.results.get() for worker in self.workers])
                    logits, ids = zip(*[self.prompts[k][i][target_idx].logits(worker.model, cand, return_ids=True) for k, worker in enumerate(self.workers)])
                    sum_target_loss = 0
                    for k, (logit, id) in enumerate(zip(logits, ids)):
                        single_target_loss = target_weight*self.prompts[k][i][target_idx].target_loss(logit, id).mean(dim=-1).to(main_device)
                        if sum_target_loss == 0: sum_target_loss = single_target_loss
                        else: sum_target_loss+=single_target_loss
                        del single_target_loss; gc.collect(); torch.cuda.empty_cache()
                    loss[j*batch_size:(j+1)*batch_size] += sum_target_loss
                    if control_weight != 0:
                        loss[j*batch_size:(j+1)*batch_size] += sum([
                            control_weight*self.prompts[k][i][target_idx].control_loss(logit, id).mean(dim=-1).to(main_device)
                            for k, (logit, id) in enumerate(zip(logits, ids))
                        ])
                        del logits, ids, sum_target_loss; gc.collect(); torch.cuda.empty_cache()
                    if verbose:
                        progress.set_description(f"loss={loss[j*batch_size:(j+1)*batch_size].min().item()/(i+1):.4f}")

            min_idx = loss.argmin()
            model_idx = min_idx // batch_size
            batch_idx = min_idx % batch_size
            next_control, cand_loss = control_cands[model_idx][batch_idx], loss[min_idx]
        
        del control_cands, loss ; gc.collect()

        print('Current length:', len(self.workers[0].tokenizer(next_control).input_ids[1:]))
        print(next_control)

        return next_control, cand_loss.item() / len(self.prompts[0]) / len(self.workers)
