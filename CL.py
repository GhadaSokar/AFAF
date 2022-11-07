
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy

class CL():
    def init_prev_masks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):                    
                    self.previous_mask[name] = torch.zeros_like(
                        param.data).to(self.device)    

    ### allocate connections based on free neurons only                    
    def create_masks(self, noParams):
        idx=0
        self.selected_nodes = {}
        self.selected_nodes_in = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):  
                    temp_mask = copy.deepcopy(self.previous_mask[name])
                    temp_mask = self.reduce(temp_mask)
                    temp_mask[temp_mask>1] = 1
                    self.mask[name] = torch.zeros_like(self.previous_mask[name])
                    # choose selected nodes layer i
                    if idx>0:
                        if (self.model.flatten_layer(name)):
                            temp_mask = self.convert_flatten_to_conv(temp_mask)
                            temp_mask[temp_mask>1] = 1
                            nodes_layer_i = np.where(selected_idx==0)
                        else:
                            nodes_layer_i = copy.deepcopy(nodes_layer_j)
                        temp_mask[:,nodes_layer_i] = 1
                    
                    if not self.model.last_layer(name):
                        ## choose selected nodes layer j
                        Free_idx_next_layer = torch.where(self.layers_free_nodes[self.model.layers_names[idx+1]]==1)
                        masked_neurons_count = (Free_idx_next_layer[0].shape[0]-self.selected_nodes_count[self.model.layers_names[idx+1]])
                        if masked_neurons_count < 0:
                            masked_neurons_count = 0
                        nodes_layer_j = np.random.choice(Free_idx_next_layer[0].cpu().numpy(), size=masked_neurons_count, replace=False)
                        temp_mask[nodes_layer_j,:] = 1
                        #remove freezed nodes layer i,j
                        temp_mask[self.layers_free_nodes[self.model.layers_names[idx+1]]==0,:] =1 
                        temp_mask[:,self.layers_free_nodes[self.model.layers_names[idx]]==0] =1 
                    else:
                        temp_mask[:,self.layers_free_nodes[self.model.layers_names[idx]]==0] = 1 
                        temp_mask[self.last_layer_active_task==0,:] = 1
                        self.selected_nodes[self.model.layers_names[idx+1]] = torch.where(self.last_layer_active_task==1)
                        self.selected_nodes_in[self.model.layers_names[idx+1]]=torch.where(self.last_layer_active_task==1)

                    idx_zeros_i,idx_zeros_j = np.where(temp_mask.to("cpu") == 0)                    
                    if(self.model.flatten_layer(name)):
                        self.selected_nodes[self.model.layers_names[idx]] = saved_idx_zeros_i
                        self.selected_nodes_in[self.model.layers_names[idx]] = list(set(idx_zeros_i))
                    else: 
                        self.selected_nodes[self.model.layers_names[idx]] = list(set(idx_zeros_j))
                        self.selected_nodes_in[self.model.layers_names[idx]] = list(set(idx_zeros_i))

                    # for flatten layer selected nodes
                    saved_idx_zeros_i = copy.deepcopy(list(set(idx_zeros_i)))
                    selected_idx = np.zeros_like(self.layers_free_nodes[self.model.layers_names[idx+1]].to("cpu").numpy())
                    selected_idx[saved_idx_zeros_i] = 1

                    new_conn_idx = np.random.choice(range(idx_zeros_i.shape[0]), size=int(noParams[idx]), replace=False)
                    if len(self.mask[name].shape)>2:
                        self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx],:,:]=1
                    else:
                        if(self.model.flatten_layer(name)):
                            conv_flatten_mask = torch.zeros_like(temp_mask)
                            conv_flatten_mask[idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]]=1
                            self.mask[name] = conv_flatten_mask.repeat_interleave(self.model.conv2_H*self.model.conv2_W,dim=1)
                        else:
                            self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]] = 1
                    idx+=1    

    ### allocate connections based on free and candidate neurons
    def create_masks_based_on_class_relation(self, selection_method_for_related_class, t2_representation, noParams):
        no_classes_in_current_task = len(self.task_labels[self.current_task])
        current_task_labels = self.task_labels[self.current_task]
        self.selected_nodes = {}
        self.selected_nodes_in = {}
        additional_selected_list = {}
        for class_idx in range(no_classes_in_current_task):
            self.candidate_neurons_per_layer = {}
            self.get_candidate_neurons_for_a_class(class_idx, selection_method_for_related_class, t2_representation)
            idx = len(self.model.layers_names) - 1
            for name, param in reversed(list(self.model.named_parameters())):
                if param.requires_grad:
                    if self.model.take_layer(name, param):  
                        temp_mask = copy.deepcopy(self.previous_mask[name])
                        temp_mask = self.reduce(temp_mask)
                        if class_idx == 0:
                            self.mask[name] = torch.zeros_like(self.previous_mask[name])
                        else:
                            temp_mask += self.reduce(self.mask[name])
                        temp_mask[temp_mask>1] = 1

                        #  masking all nodes except the selected 
                        #  for layer i the selected nodes are the selected nodes for layer j of previous layer
                        if idx<(len(self.model.layers_names) - 1):  
                            nodes_layer_j=copy.deepcopy(nodes_layer_i)
                            temp_mask[nodes_layer_j,:] = 1
                        
                        if not self.model.last_layer(name):
                            ## Flatten layer
                            if (self.model.flatten_layer(name)):
                                temp_mask = self.convert_flatten_to_conv(temp_mask)
                                temp_mask[temp_mask>1] = 1 
                            if idx > 1:
                                # select some additional nodes for new task
                                if class_idx==0:
                                    additional_selected_list[self.model.layers_names[idx-1]] = torch.zeros_like(self.candidate_neurons_per_layer[self.model.layers_names[idx-1]])
                                    Free_idx_current_layer = torch.where((self.layers_free_nodes[self.model.layers_names[idx-1]]==1).to(self.device)& (self.candidate_neurons_per_layer[self.model.layers_names[idx-1]]==0).to(self.device))
                                    no_additional_selected = min(self.additional_selected_nodes[name], Free_idx_current_layer[0].cpu().numpy().shape[0])
                                    allow  = np.random.choice(Free_idx_current_layer[0].cpu().numpy(), size=no_additional_selected, replace=False)
                                    additional_selected_list[self.model.layers_names[idx-1]][allow] = 1
                                print("size of candidate",(self.candidate_neurons_per_layer[self.model.layers_names[idx-1]]==1).sum())
                                print("candidate neurons",torch.where(self.candidate_neurons_per_layer[self.model.layers_names[idx-1]]==1))
                                nodes_layer_i = torch.where((self.candidate_neurons_per_layer[self.model.layers_names[idx-1]]==0) & (additional_selected_list[self.model.layers_names[idx-1]]==0))[0]
                                print("size of node_layer_i", nodes_layer_i.shape)
                                #temp_mask[:,self.candidate_neurons_per_layer[self.model.layers_names[idx-1]]==0] = 1
                                temp_mask[:,nodes_layer_i] = 1

                            # if we  will take candidate in last layer, instead of if not self.model.last_layer >> if True
                            if self.model.last_layer(name):
                                self.last_layer_active_task = torch.zeros(
                                self.model.num_classes).to(self.device)
                                self.last_layer_active_task[self.task_labels[self.current_task][class_idx]] = 1 
                                temp_mask[self.last_layer_active_task==0,:] = 1
                            else:
                                temp_mask[self.layers_free_nodes[self.model.layers_names[idx]]==0,:] = 1 

                        else:
                            temp_mask[:,self.layers_free_nodes[self.model.layers_names[idx-1]]==0] = 1                             
                            Free_idx_current_layer = torch.where(self.layers_free_nodes[self.model.layers_names[idx-1]]==1)

                            if class_idx == 0: 
                                masked_neurons_count = Free_idx_current_layer[0].shape[0]-self.selected_nodes_count[self.model.layers_names[idx-1]]
                                if masked_neurons_count < 0:
                                    masked_neurons_count = 0
                                nodes_layer_i = np.random.choice(Free_idx_current_layer[0].cpu().numpy(), size=masked_neurons_count, replace=False)
                                saved_node_layer_u_for_last_layer = copy.deepcopy(nodes_layer_i)
                            else:
                                nodes_layer_i = copy.deepcopy(saved_node_layer_u_for_last_layer)
                            temp_mask[:,nodes_layer_i] = 1

                            self.last_layer_active_task = torch.zeros(
                            self.model.num_classes).to(self.device)
                            self.last_layer_active_task[self.task_labels[self.current_task][class_idx]] = 1 
                            temp_mask[self.last_layer_active_task==0,:] = 1
                            if class_idx == 0:
                                self.selected_nodes[self.model.layers_names[idx]] = (torch.where(self.last_layer_active_task==1)[0]).tolist()
                                self.selected_nodes_in[self.model.layers_names[idx]] = (torch.where(self.last_layer_active_task==1)[0]).tolist()
                            else:
                                self.selected_nodes[self.model.layers_names[idx]]+= (torch.where(self.last_layer_active_task==1)[0]).tolist()
                                self.selected_nodes_in[self.model.layers_names[idx]]+= (torch.where(self.last_layer_active_task==1)[0]).tolist()

                        # the remaining elements is temp_mask is the places where we can allocate connection for the current task
                        idx_zeros_i,idx_zeros_j = np.where(temp_mask.to("cpu") == 0)   
                        if class_idx == 0:
                            self.selected_nodes[self.model.layers_names[idx-1]]=list(set(idx_zeros_j))
                            self.selected_nodes_in[self.model.layers_names[idx-1]]=list(set(idx_zeros_i))
                        else:
                            self.selected_nodes[self.model.layers_names[idx-1]]+=list(set(idx_zeros_j))
                            self.selected_nodes[self.model.layers_names[idx-1]]=list(set(self.selected_nodes[self.model.layers_names[idx-1]]))

                            self.selected_nodes_in[self.model.layers_names[idx-1]]+=list(set(idx_zeros_i))
                            self.selected_nodes_in[self.model.layers_names[idx-1]]=list(set(self.selected_nodes_in[self.model.layers_names[idx-1]]))

                            print("len of selected nodes",len(self.selected_nodes[self.model.layers_names[idx-1]]))
 
                        no_param_per_class = noParams[idx-1]/no_classes_in_current_task
                        print('idx', idx)
                        print('size of free', idx_zeros_i.shape[0])
                        print('required no of samples',no_param_per_class )
                        new_conn_idx = np.random.choice(range(idx_zeros_i.shape[0]), size=int(no_param_per_class),replace=False)
                        if no_param_per_class>0:
                            if len(self.mask[name].shape)>2:
                                self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx],:,:] = 1
                            else:
                                if(self.model.flatten_layer(name)):
                                    conv_flatten_mask = torch.zeros_like(temp_mask)
                                    conv_flatten_mask[idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]] = 1
                                    self.mask[name] += conv_flatten_mask.repeat_interleave(self.model.conv2_H*self.model.conv2_W,dim=1)
                                else: 
                                    self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]] = 1
                        idx-=1  
                         
        self.last_layer_active_task = torch.zeros(
        self.model.num_classes).to(self.device)
        self.last_layer_active_task[self.task_labels[self.current_task]] = 1 

    ## get candidate for each layer
    def get_candidate_neurons_for_a_class(self, current_class_id, selection_method_for_related_class, task_representation):        
        idx = 0 
        name_idx = len(self.model.layers_names) -3
        for name, param in reversed(list(self.model.named_parameters())):
            if name_idx>=0:
                if param.requires_grad:
                    if self.model.take_layer(name, param):
                        if (self.model.flatten_layer(name)):
                            self.candidate_neurons_per_layer[name] = torch.zeros(
                                param.shape[1]//(self.model.conv2_W*self.model.conv2_H)).to(self.device)                            
                        else: 
                            self.candidate_neurons_per_layer[name] = torch.zeros(
                                param.shape[1]).to(self.device)
                        idx+=1                           
                        connected_neurons = copy.deepcopy(self.get_connected_based_on_representation(current_class_id, task_representation, int(self.reuse_neuron_count[self.model.layers_names[name_idx+1]]), selection_method_for_related_class, idx))
                        name_idx-=1
                        self.candidate_neurons_per_layer[name][connected_neurons] = 1


    ## get candidate based on activation
    def get_connected_based_on_representation(self, current_class_id, task_representation, no_reuse_neuron, selection_method_for_related_class, idx):
        current_layer_idx = len(task_representation) - idx
        if selection_method_for_related_class=='leastrelated':
            sort_descent = False
        elif selection_method_for_related_class=='mostrelated':
            sort_descent = True    
        current_layer_representation = task_representation[current_layer_idx][current_class_id]
        
        if selection_method_for_related_class=='random':
            selected_neurons = np.random.choice(np.arange(0,current_layer_representation.shape[0]), size=no_reuse_neuron, replace=False)
        elif selection_method_for_related_class=='mostrelated':
            selected_neurons = torch.argsort(current_layer_representation, descending=sort_descent)[0:no_reuse_neuron]
        elif selection_method_for_related_class=='leastrelated':
            tmp_selected_neurons = torch.argsort(current_layer_representation, descending=sort_descent)
            num_zeros=(current_layer_representation==0).sum()
            selected_neurons = tmp_selected_neurons[num_zeros:no_reuse_neuron+num_zeros]

        #print(current_layer_representation[selected_neurons])

        return selected_neurons


    def __init__(self, device, freezed_nodes_count_perlayer, num_selected_nodes, task_labels, model, num_additional_selected_nodes, no_neurons_reused_from_previous):
        self.model = model
        self.device = device
        self.replace_percentage = 0.2
        print("self.replace_percentage.............", self.replace_percentage)
        self.inf = 99999
        self.mask = {}
        self.previous_mask= {}
        self.task_labels = task_labels
        self.current_task = 0
        self.freezed_nodes_count = freezed_nodes_count_perlayer
        self.num_additional_selected_nodes = num_additional_selected_nodes
        self.num_selected_nodes = num_selected_nodes
        self.no_neurons_reused_from_previous = no_neurons_reused_from_previous
        self.init_freezed_nodes()
        self.init_prev_masks()
        self.create_masks(self.model.noParams)
        self.first_mask_update = True 

    def reduce(self, tensor):
        if(len(tensor.shape) == 2):
            return tensor
        return tensor.sum(dim=(2, 3))
    def convert_flatten_to_conv(self,mask):
        nodes_count = mask.shape[1]//(self.model.conv2_W*self.model.conv2_H)
        dims = (mask.shape[0], nodes_count, self.model.conv2_W, self.model.conv2_H)
        return self.reduce(mask.reshape(dims))

    ### drop connections for DST training. This function is adpoted from the official code of the SpaceNet paper
    def remove(self):
        self.removed_mask = {}
        self.replace_count = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    current_mask=copy.deepcopy(self.mask[name])
                    importance = copy.deepcopy(self.weights_importance[name]).to(self.device)
                    if(self.model.flatten_layer(name)):
                        current_mask=self.convert_flatten_to_conv(current_mask)
                        importance = self.convert_flatten_to_conv(importance)
                        current_mask[current_mask>0]=1
                    reduced_mask=self.reduce(current_mask)
                    reduced_mask[reduced_mask>0]=1
                    total = torch.sum(reduced_mask)
                    replace_count = int(total*self.replace_percentage)
                    self.replace_count[name] = replace_count
                    if self.replace_count[name] == 0:
                        continue
                    importance += ((1-current_mask)*self.inf)
                    reduced_importance=self.reduce(abs(importance))
                    reduced_importance = reduced_importance.flatten()
                    idx = np.argpartition(reduced_importance.to("cpu"), replace_count)
                    removed_mask = torch.zeros_like(reduced_importance).to(self.device)
                    removed_mask[idx[:replace_count]] = 1
                    removed_mask = removed_mask.reshape(
                        reduced_mask.shape)
                    self.removed_mask[name] = copy.deepcopy(
                        removed_mask).to(self.device)

    ### add connections for DST training. This function is adpoted from the official code of the SpaceNet paper
    def add(self):
        for idx in range(len(self.model.layers_names)-2):
            name = self.model.layers_names[idx]
            if self.replace_count[name] == 0:
                   continue

            nxt_name = self.model.layers_names[idx+1]
            layer_importnace = torch.mm(self.layers_importnace[nxt_name].reshape(self.layers_importnace[nxt_name].shape[0], 1),
                                        self.layers_importnace[name].reshape(self.layers_importnace[name].shape[0], 1).T)
            not_selected_nodes=torch.ones_like(layer_importnace)
            not_selected_nodes[:,self.selected_nodes[name]]-= 1
            not_selected_nodes[self.selected_nodes_in[name],:]-= 1
            not_selected_nodes[not_selected_nodes==0]=1
            not_selected_nodes[not_selected_nodes==-1]=0
            reduced_mask = self.reduce(self.mask[name]+self.previous_mask[name])
            reduced_mask[reduced_mask>0]=1
            if(self.model.flatten_layer(name)):
                reduced_mask=reduced_mask[:,::self.model.conv2_H*self.model.conv2_W]
            #- removed_mask to allow add cnnections in the same previous place
            reduced_mask = reduced_mask-self.removed_mask[name]
            layer_importnace[reduced_mask==1] = -self.inf
            layer_importnace[not_selected_nodes==1]=-self.inf
            layer_importnace = -layer_importnace.flatten()
            idx_add = np.argpartition(layer_importnace.to(
                "cpu"), self.replace_count[name])
            
            assert(torch.max(layer_importnace[idx_add[:self.replace_count[name]]])<self.inf) 

            added_mask = torch.zeros_like(layer_importnace).to(self.device)
            added_mask[idx_add[:self.replace_count[name]]] = 1
            added_mask = added_mask.reshape(reduced_mask.shape)
            if(self.model.flatten_layer(name)):
                added_mask=added_mask.repeat_interleave(self.model.conv2_H*self.model.conv2_W,dim=1)
                self.removed_mask[name]=self.removed_mask[name].repeat_interleave(self.model.conv2_H*self.model.conv2_W,dim=1)
            self.mask[name][self.removed_mask[name]==1]=0
            self.mask[name][added_mask==1]=1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    param.data = param.data*(self.mask[name]+self.previous_mask[name]).to(self.device)
        
    def set_init_network_weight(self):
        self.init_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):                   
                    self.init_weights[name] = copy.deepcopy(param.data)
                    param.data = param.data*self.mask[name].to(self.device)

    def set_old_weight(self):
        self.old_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.old_weights[name] = copy.deepcopy(param.data)

    def apply_mask_on_grad(self):
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param): #'weight' in name: #
                    param.grad = param.grad*self.mask[name].to(self.device)
                    idx+=1
                elif 'bias' in name:
                    param.grad = param.grad*self.layers_free_nodes[self.model.layers_names[idx]].to(self.device)

    def reset_importance(self):
        self.weights_importance = {}
        self.layers_importnace = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.weights_importance[name] = torch.zeros_like(
                       param.data)
                    self.layers_importnace[name] = torch.zeros(
                        param.shape[1]).to(self.device)
                    if(self.model.flatten_layer(name)):
                        self.layers_importnace[name] = torch.zeros(self.layers_free_nodes[name].shape[0]).to(self.device)                     

    def reset_w_importance(self):
        self.weights_importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.weights_importance[name] = torch.zeros_like(
                       param.data)

    ## calculate neuron importance based on importance of outgoing connections
    def calculate_importance(self):
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.weights_importance[name] += abs((param.data-self.old_weights[name])*param.grad*self.mask[name])
                    layer_importnace = torch.sum(self.weights_importance[name], dim=0).squeeze()
                    if(len(layer_importnace.shape) > 1):
                        layer_importnace = abs(layer_importnace)
                        layer_importnace = layer_importnace.sum(
                            dim=(-1, -2)).squeeze().to(self.device)
                    if(self.model.flatten_layer(name)):
                        layer_importnace=layer_importnace.reshape((self.layers_free_nodes[name].shape[0],self.model.conv2_H,self.model.conv2_W,1))
                        layer_importnace=self.reduce(layer_importnace).sum(axis=1)
                    self.layers_importnace[name] += layer_importnace
    # For CL
    def recover_old_task_weight(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    param.data[self.previous_mask[name]==1]  = self.old_weights[name][self.previous_mask[name]==1] 

    def retain_last_layer_and_init_next_task_weights(self):
        for name, param in self.model.named_parameters():
            if self.model.take_layer(name,param):
                if self.model.last_layer(name):
                    self.init_weights[name][self.task_labels[self.current_task-1],:] = param.data[self.task_labels[self.current_task-1],:]
                    param.data=torch.zeros_like(self.init_weights[name])

                param.data[self.mask[name]==1]=self.init_weights[name][self.mask[name]==1]

    ## return output weights of previous tasks 
    def set_classifer_to_all_learned_tasks(self):
        for name, param in self.model.named_parameters():
            if self.model.take_layer(name,param):
                if self.model.last_layer(name):
                    for i in range(self.current_task):
                        param.data[self.task_labels[i],:] = self.init_weights[name][self.task_labels[i],:] 
                        
    def init_freezed_nodes(self):
        self.layers_free_nodes = {}
        self.num_freezed_nodes = {}
        self.selected_nodes_count = {}
        self.additional_selected_nodes = {}
        self.reuse_neuron_count = {}
        i=0
        last_dim=0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    if len(self.num_selected_nodes)>0:
                        select_dim = self.num_selected_nodes[i]
                        add_nodes = self.num_additional_selected_nodes[i]
                    else:
                        select_dim = param.shape[1]
                        add_nodes = param.shape[1]
                    dim = param.shape[1]
                    if(self.model.flatten_layer(name)):
                        dim = last_dim
                        select_dim = last_select_dim
                    self.layers_free_nodes[name] = torch.ones(dim).to(self.device)
                    self.num_freezed_nodes[name] = self.freezed_nodes_count[i]
                    self.selected_nodes_count[name] = select_dim
                    self.additional_selected_nodes[name] = add_nodes
                    self.reuse_neuron_count[name] = self.no_neurons_reused_from_previous[i]
                    i+=1

                    last_dim = param.shape[0]
                    last_select_dim = param.shape[0]
                    if len(self.num_selected_nodes)>0:
                        last_select_dim = self.num_selected_nodes[i]
                    
        self.layers_free_nodes[name]=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.layers_free_nodes[name][self.task_labels[self.current_task]]=1                
        self.last_layer_active_task=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.last_layer_active_task[self.task_labels[self.current_task]] = 1 

    def update_freezed_nodes(self):
        idx=0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    if idx>0:
                        layeridxImp_tmp=np.argsort(self.layers_importnace[name].to("cpu").numpy())[::-1]
                        current_super_set_nodes = set(self.selected_nodes[self.model.layers_names[idx]])
                        if len(current_super_set_nodes)>0:
                            nodes_to_select = []
                            
                            for node_id in layeridxImp_tmp:
                                if node_id in current_super_set_nodes:
                                    #if self.layers_importnace[name][node_id]>0:
                                    nodes_to_select.append(node_id)
                            layeridxImp_tmp = np.array(nodes_to_select)

                            tmp_layer=self.layers_free_nodes[name].to("cpu").numpy()
                            tmp_layer[layeridxImp_tmp[:self.num_freezed_nodes[name]]] = 0
                            self.layers_free_nodes[name] = torch.from_numpy(tmp_layer)
                            
                    idx+=1
            
        self.layers_free_nodes[name]=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.layers_free_nodes[name][self.task_labels[self.current_task+1]]=1
        self.last_layer_active_task=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.last_layer_active_task[self.task_labels[self.current_task+1]] = 1

    def prepare_next_task(self, selection_method_for_related_class, enable_reuse, t2_representation=None):
        noParams = copy.copy(self.model.noParams)
        print("enable_reuse......", enable_reuse)
        print("lreuse....",self.model.l_reuse)
        if enable_reuse: 
            for i in range((len(self.model.noParams))):
                if i < self.model.l_reuse:
                    noParams [i] = 0
        print("parameters for task {} equals {}".format(self.current_task+1, noParams))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.previous_mask[name] += self.mask[name]
    
        self.update_freezed_nodes()
        self.current_task+=1       
        if not enable_reuse: #self.current_task ==0:   
            self.create_masks(noParams)
        else:
            self.create_masks_based_on_class_relation(selection_method_for_related_class, t2_representation, noParams)
        self.retain_last_layer_and_init_next_task_weights()

    def print_nodes(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    print(f'freezed nodes layer {name} ')
                    print(torch.where(self.layers_free_nodes[name]==0))


