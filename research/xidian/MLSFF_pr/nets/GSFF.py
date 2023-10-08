import mindspore
import mindspore.nn as nn
from functools import reduce
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.ops import functional as F
from mindspore.common.initializer import Zero
reshape = ops.Reshape()
expand_dims = mindspore.ops.ExpandDims()
cast = ops.Cast()

def build_embeding_layer(vocab_size, dim, drop_prob):
    embed = nn.SequentialCell([nn.Embedding(vocab_size, dim),
                          nn.ReLU()])
    return embed

class Attention(nn.Cell):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size
        self.tanh = nn.Tanh()
        self.h2att = nn.Dense(in_channels=self.rnn_size, out_channels=self.att_hid_size)
        self.alpha_net = nn.Dense(in_channels=self.att_hid_size, out_channels=1)

    def construct(self, h, att_feats, p_att_feats):
        att_h = self.h2att(h)                       
        att_h = expand_dims(att_h,1).expand_as(p_att_feats)
        dot = p_att_feats + att_h
        dot = self.alpha_net(dot)
        dot = ops.squeeze(dot)
        weight = mindspore.ops.Softmax(axis=1)(dot) 
        att_res = ops.squeeze(ops.matmul(expand_dims(weight,1), att_feats),1)

        return att_res
    
    

class TopDownCore(nn.Cell):
    def __init__(self, opt, use_maxout=False):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = opt.drop_prob_lm

        self.att_lstm = nn.LSTM(opt.input_encoding_size + opt.rnn_size * 2, opt.rnn_size) 
        self.lang_lstm = nn.LSTM(opt.rnn_size * 2, opt.rnn_size) 
        self.attention_attr = Attention(opt)

    def construct(self, xt, fc_feats, att_feats, p_att_feats, state, attr_feats, p_attr_feats):
        prev_h = state[0][1]
        att_lstm_input = ops.Concat(1)([prev_h, fc_feats, xt])
        att_lstm_input = expand_dims(att_lstm_input,0)
        h_att_pre = expand_dims(state[0][0],0)
        c_att_pre = expand_dims(state[1][0],0)
        _, (h_att, c_att) = self.att_lstm(att_lstm_input, (h_att_pre, c_att_pre))
        h_att = ops.squeeze(h_att)
        c_att = ops.squeeze(c_att)
        att_attr = self.attention_attr(h_att, attr_feats, p_attr_feats)
        lang_lstm_input = ops.Concat(1)([att_attr, h_att])
        lang_lstm_input = expand_dims(lang_lstm_input,0)
        h_lang_pre = expand_dims(state[0][1],0)
        c_lang_pre = expand_dims(state[1][1],0)
        _, (h_lang, c_lang) = self.lang_lstm(lang_lstm_input, (h_lang_pre, c_lang_pre))
        h_lang = ops.squeeze(h_lang)
        c_lang = ops.squeeze(c_lang)
        output = h_lang
        state = (mindspore.ops.stack([h_att, h_lang]), mindspore.ops.stack([c_att, c_lang]))
        return output, state


class GSFF(nn.Cell):
    def __init__(self, opt):
        super(GSFF, self).__init__()
        self.num_layers = 2
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.num_attrs = opt.num_attr
        self.sg_label_embed_size = opt.sg_label_embed_size
        self.use_num_attr=opt.use_num_attr
        self.ss_prob = 0.0 
        self.embed = nn.SequentialCell([nn.Embedding(self.vocab_size + 1, self.input_encoding_size),
                                nn.ReLU()])
        self.fc_embed = nn.SequentialCell([nn.Dense(in_channels=self.fc_feat_size, out_channels=self.rnn_size),nn.ReLU()])
        self.att_embed = nn.SequentialCell([nn.Dense(in_channels=self.att_feat_size, out_channels=self.rnn_size),nn.ReLU()])
        self.attr_embed = build_embeding_layer(self.num_attrs, self.sg_label_embed_size, self.drop_prob_lm)
        self.attr_proj = nn.SequentialCell(*[nn.Dense(in_channels=self.sg_label_embed_size*self.use_num_attr, out_channels=self.rnn_size),
                                         nn.ReLU()])
        self.fusion_attr = nn.SequentialCell([nn.Dense(in_channels=self.rnn_size * 2, out_channels=self.rnn_size), nn.ReLU()])
        self.ctx2att_attr = nn.Dense(in_channels=self.rnn_size, out_channels=self.att_hid_size)


        self.logit_layers = getattr(opt, 'logit_layers', 1)
        self.logit = nn.Dense(in_channels=self.rnn_size, out_channels=self.vocab_size + 1)

        self.ctx2att = nn.Dense(in_channels=self.rnn_size, out_channels=self.att_hid_size)
        self.core = TopDownCore(opt)
        
    def global_fusion_operation(self, attr_labels, att_feats):
        tmp_attr = attr_labels[:,:,0:(self.use_num_attr)]
        tmp_attr = cast(tmp_attr, mindspore.int32)
        attr_vecs = self.attr_embed(tmp_attr)
        attr_vecs = reshape(attr_vecs,(mindspore.ops.shape(attr_vecs)[0], mindspore.ops.shape(attr_vecs)[1], mindspore.ops.shape(attr_vecs)[2]*mindspore.ops.shape(attr_vecs)[3]))
        attr_vecs = self.attr_proj(attr_vecs)
        B, No = attr_vecs.shape[:2] 
        attr_vecs = reshape(attr_vecs, (B,No, attr_vecs.shape[2]))
        attr_vecs = attr_vecs.repeat(att_feats.shape[1],axis=1)
        attr_vecs = self.fusion_attr(ops.Concat(2)([att_feats, attr_vecs])) + attr_vecs
        return attr_vecs
    
    def _prepare_feature(self, fc_feats, att_feats,attr_labels):
        fc_feats = self.fc_embed(fc_feats)
        att_feats = self.att_embed(att_feats)
        pp_att_feats = self.ctx2att(att_feats) 
        p_attr_feats=[]
        if self.use_num_attr>0:
            attr_feats = self.global_fusion_operation(attr_labels, att_feats)
            p_attr_feats = self.ctx2att_attr(attr_feats)
        else:
            attr_feats = att_feats
            p_attr_feats = pp_att_feats
        p_fc_feats, p_att_feats = fc_feats, att_feats 
        return p_fc_feats, p_att_feats, pp_att_feats,attr_feats,p_attr_feats

    def construct(self, inputs,labels):
        fc_feats = inputs[0]
        att_feats = inputs[1]
        attr_labels  = inputs[2]
        seq = labels
        batch_size = ops.Shape()(fc_feats)[0]
        state = (mindspore.ops.Zeros()((self.num_layers, batch_size, self.rnn_size),fc_feats.dtype),
                mindspore.ops.Zeros()((self.num_layers, batch_size, self.rnn_size),fc_feats.dtype))
        it = mindspore.ops.Zeros()((batch_size),fc_feats.dtype)
        p_fc_feats, p_att_feats, pp_att_feats, attr_feats, p_attr_feats= self._prepare_feature(fc_feats, att_feats,attr_labels)


        for i in range(ops.Shape()(seq)[1] - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:
                sample_prob = ops.uniform((batch_size),0,1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].copy()
                else:
                    print("unfinished!")
            else:
                it = seq[:, i].copy()          
            if i >= 1 and seq[:, i].astype(mindspore.float32).sum() == 0:
                break
            output, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, attr_feats, p_attr_feats, state)
            output = output.reshape(output.shape[0],1,output.shape[1])
            if i ==0:
                outputs = output
            else:
                outputs = ops.Concat(1)((outputs, output))
        return outputs

    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, attr_feats, p_attr_feats, state):
        it = cast(it, mindspore.int32)
        xt = self.embed(it)
        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, attr_feats, p_attr_feats)
        logprobs = ops.LogSoftmax(axis=1)(self.logit(output))
        return logprobs, state

    def _sample_beam(self, fc_feats, att_feats,attr_labels, opt={}):
        beam_size = opt.get('beam_size', 1)
        batch_size = mindspore.ops.shape(fc_feats)[0]
        p_fc_feats, p_att_feats, pp_att_feats,attr_feats, p_attr_feats= self._prepare_feature(fc_feats, att_feats,attr_labels)
        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = mnp.zeros((self.seq_length, batch_size),dtype=p_fc_feats.dtype)
        seqLogprobs = mnp.zeros((self.seq_length, batch_size),dtype=p_fc_feats.dtype)
        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            tmp_fc_feats = mnp.repeat(p_fc_feats[k:k+1], beam_size, 0)
            tmp_att_feats = mnp.repeat(p_att_feats[k:k+1],beam_size,0)
            tmp_p_att_feats = mnp.repeat(pp_att_feats[k:k+1],beam_size,0)
            tmp_attr_feats = mnp.repeat(attr_feats[k:k+1],beam_size,0)
            tmp_p_attr_feats = mnp.repeat(p_attr_feats[k:k+1],beam_size,0)
            state = (mindspore.ops.Zeros()((self.num_layers, beam_size, self.rnn_size),p_fc_feats.dtype),
                mindspore.ops.Zeros()((self.num_layers, beam_size, self.rnn_size),p_fc_feats.dtype))
            for t in range(1):
                if t == 0: 
                    it = mindspore.ops.Zeros()((beam_size),mindspore.int32)
                logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_attr_feats, tmp_p_attr_feats, state)
            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats,tmp_attr_feats,tmp_p_attr_feats, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] 
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        seq = seq.astype(mindspore.int32)
        return ops.transpose(seq,(1, 0)), ops.transpose(seqLogprobs,(1, 0))

    def sample(self, fc_feats, att_feats, attr_labels, opt={}):
        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, attr_labels, opt)
        batch_size = ops.Shape()(fc_feats)[0]
        state = (mindspore.ops.Zeros()((self.num_layers, batch_size, self.rnn_size),fc_feats.dtype),
                mindspore.ops.Zeros()((self.num_layers, batch_size, self.rnn_size),fc_feats.dtype))
        p_fc_feats, p_att_feats, pp_att_feats, attr_feats, p_attr_feats = self._prepare_feature(fc_feats, att_feats,attr_labels)
        for t in range(self.seq_length + 1):
            if t == 0:
                it = mnp.zeros((batch_size), mindspore.int32)
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats,attr_feats, p_attr_feats, state)
            if decoding_constraint and t > 0:
                tmp = mnp.zeros((mindspore.ops.shape(logprobs)),logprobs.dtype)
                tmp = F.tensor_scatter_elements(tmp, expand_dims(seq[:,t-1],1),float('-inf'),1)
                logprobs = logprobs + tmp

            if t == self.seq_length: 
                break
            sampleLogprobs = logprobs.max(1)
            it = logprobs.argmax(1)
            it = cast(it, mindspore.int32)
            if t == 0:
                unfinished = it > 0
            else:
                unfinished = unfinished * (it > 0)
            unfinished = cast(unfinished, it.dtype)
            it = it * unfinished
            if t == 0:
                seq = it.reshape(it.shape[0],1)
                seqLogprobs = sampleLogprobs.reshape(sampleLogprobs.shape[0],1)
            else:
                seq = ops.Concat(1)((seq,it.reshape(it.shape[0],1)))
                seqLogprob = ops.Concat(1)((seqLogprobs,sampleLogprobs.reshape(sampleLogprobs.shape[0],1)))
            seq = seq.astype(mindspore.int32)
            if unfinished.astype(mindspore.float32).sum() == 0:
                seq=ops.Pad(((0,0),(0,self.seq_length-seq.shape[1])))(seq)
                break

        return seq, seqLogprobs     
    
    def beam_search(self, init_state, init_logprobs, *args, **kwargs):
        def add_diversity(beam_seq_table, logprobsf, t, divm, diversity_lambda, bdash):
            local_time = t - divm
            unaug_logprobsf = logprobsf.copy()
            for prev_choice in range(divm):
                prev_decisions = beam_seq_table[prev_choice][local_time]
                for sub_beam in range(bdash):
                    for prev_labels in range(bdash):
                        logprobsf[sub_beam][prev_decisions[prev_labels]] = logprobsf[sub_beam][prev_decisions[prev_labels]] - diversity_lambda
            return unaug_logprobsf
 

        def beam_step(logprobsf, unaug_logprobsf, beam_size, t, beam_seq, beam_seq_logprobs, beam_logprobs_sum, state):
            ys,ix = ops.Sort(1,True)(logprobsf)
            candidates = []
            cols = min(beam_size, mindspore.ops.shape(ys)[1])
            rows = beam_size
            if t == 0:
                rows = 1
            for c in range(cols): 
                for q in range(rows):
                    #local_logprob = ys[q,c].item()
                    local_logprob = ys[q,c].reshape([1]).item()
                    candidate_logprob = beam_logprobs_sum[q] + local_logprob
                    local_unaug_logprob = unaug_logprobsf[q,ix[q,c]]
                    candidates.append({'c':ix[q,c], 'q':q, 'p':candidate_logprob, 'r':local_unaug_logprob})
            candidates = sorted(candidates,  key=lambda x: -x['p'])
            
            new_state = [_.copy() for _ in state]
            if t >= 1:
                beam_seq_prev = beam_seq[:t].copy()
                beam_seq_logprobs_prev = beam_seq_logprobs[:t].copy()
            for vix in range(beam_size):
                v = candidates[vix]
                if t >= 1:
                    beam_seq[:t, vix] = beam_seq_prev[:, v['q']]
                    beam_seq_logprobs[:t, vix] = beam_seq_logprobs_prev[:, v['q']]
                for state_ix in range(len(new_state)):
                    new_state[state_ix][:, vix] = state[state_ix][:, v['q']] 
                beam_seq[t, vix] = v['c'] 
                beam_seq_logprobs[t, vix] = v['r'] 
                beam_logprobs_sum[vix] = v['p'] 
            state = new_state
            #beam_seq = beam_seq.astype(mindspore.int32)
            return beam_seq,beam_seq_logprobs,beam_logprobs_sum,state,candidates

        opt = kwargs['opt']
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        diversity_lambda = opt.get('diversity_lambda', 0.5)
        decoding_constraint = opt.get('decoding_constraint', 0)
        split_op0 = ops.Split(0, group_size)
        split_op2 = ops.Split(2, group_size)
        max_ppl = opt.get('max_ppl', 0)
        bdash = beam_size // group_size 
        beam_seq_table = [mnp.zeros((self.seq_length, bdash),dtype=mindspore.float32) for _ in range(group_size)]
        beam_seq_logprobs_table = [mnp.zeros((self.seq_length, bdash),dtype = mindspore.float32) for _ in range(group_size)]
        beam_logprobs_sum_table = [mnp.zeros((bdash)) for _ in range(group_size)]
        unstack = ops.Unstack()
        done_beams_table = [[] for _ in range(group_size)]
        state_table = [list(unstack(_)) for _ in split_op2(mindspore.ops.stack(init_state))]
        logprobs_table = list(split_op0(init_logprobs))


        args = list(args)
        args = [split_op0(_) if _ is not None else [None]*group_size for _ in args]
        args = [[args[i][j] for i in range(len(args))] for j in range(group_size)]

        for t in range(self.seq_length + group_size - 1):
            for divm in range(group_size): 
                if t >= divm and t <= self.seq_length + divm - 1:
                    logprobsf = logprobs_table[divm]
                    if decoding_constraint and t-divm > 0:
                        # logprobsf.scatter_(1, expand_dims(beam_seq_table[divm][t-divm-1], 1).cuda(), float('-inf'))
                        logprobsf = F.tensor_scatter_elements(logprobsf, expand_dims(beam_seq_table[divm][t-divm-1], 1),float('-inf'),1)
                    logprobsf[:,mindspore.ops.shape(logprobsf)[1]-1] = logprobsf[:, mindspore.ops.shape(logprobsf)[1]-1] - 1000  
                    unaug_logprobsf = add_diversity(beam_seq_table,logprobsf,t,divm,diversity_lambda,bdash)
                    beam_seq_table[divm],\
                    beam_seq_logprobs_table[divm],\
                    beam_logprobs_sum_table[divm],\
                    state_table[divm],\
                    candidates_divm = beam_step(logprobsf,
                                                unaug_logprobsf,
                                                bdash,
                                                t-divm,
                                                beam_seq_table[divm],
                                                beam_seq_logprobs_table[divm],
                                                beam_logprobs_sum_table[divm],
                                                state_table[divm])

                    for vix in range(bdash):
                        if beam_seq_table[divm][t-divm,vix] == 0 or t == self.seq_length + divm - 1:
                            final_beam = {
                                'seq': beam_seq_table[divm][:, vix].copy(), 
                                'logps': beam_seq_logprobs_table[divm][:, vix].copy(),
                                'unaug_p': beam_seq_logprobs_table[divm][:, vix].sum().reshape([1]).item(),
                                'p': beam_logprobs_sum_table[divm][vix].reshape([1]).item()
                            }
                            if max_ppl:
                                final_beam['p'] = final_beam['p'] / (t-divm+1)
                            done_beams_table[divm].append(final_beam)
                            beam_logprobs_sum_table[divm][vix] = -1000

                    
                    it = beam_seq_table[divm][t-divm]
                    logprobs_table[divm], state_table[divm] = self.get_logprobs_state(it, *(args[divm] + [state_table[divm]]))

        done_beams_table = [sorted(done_beams_table[i], key=lambda x: -x['p'])[:bdash] for i in range(group_size)]
        done_beams = reduce(lambda a,b:a+b, done_beams_table)
        return done_beams
