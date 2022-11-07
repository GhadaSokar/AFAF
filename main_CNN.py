import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import numpy as np
import argparse
import random
from sklearn.metrics import confusion_matrix
import time
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import models_CNN
import CL
import utils_CNN

### training one task in the CL sequence
def train_task(args, cl, train_loader, val_loader, test_loader, optimizer, criterion, device, task_idx, task_labels):
    test_acc_along_training_CIL = []
    train_acc_along_training_CIL = []
    test_acc_along_training_TIL = []
    train_acc_along_training_TIL = []
    for epoch in range(args.epochs):
        cl.model.train()
        t0=time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device,dtype=torch.int64)
            # zero the parameter gradients   
            optimizer.zero_grad()
            # save weights
            cl.set_old_weight()
            outputs, _,_,_,_,_ = cl.model(data)
            # Computes loss
            loss = criterion(outputs, target)
            #compute gradient 
            loss.backward()
            cl.apply_mask_on_grad()
            optimizer.step()
            cl.calculate_importance()
            cl.recover_old_task_weight()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                100. * batch_idx / len(train_loader), loss.item()))
                sys.stdout.flush()    
        print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
        if epoch < args.epochs-1: 
            #### drop and grow cycle from the SpaceNet algorithm #####
            cl.remove()
            cl.add()

    return train_acc_along_training_CIL, test_acc_along_training_CIL, train_acc_along_training_TIL, test_acc_along_training_TIL

def main():
    parser = argparse.ArgumentParser(description='AFAF algorithm')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=10, metavar='S',
                         help='random seed (default: 10)')
    parser.add_argument('--benchmark', type=str, default= 'CIFAR10', 
                        help= 'Options: CIFAR10, CIFAR100, mix')
    parser.add_argument('--num_classes_per_task', type=int, default=2,
                        help='number of classes in each task (default: 2 for CIFAR)')
    parser.add_argument('--num_tasks', type=int, default=5,
                        help='number of tasks')
    parser.add_argument('--class_order', default='0,1,2,3,4,5,6,7,8,9', 
                        help='new order for classes, None if orginial order required') 
    parser.add_argument('--epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0, metavar='M',
                        help='SGD momentum (default: 0.0)')
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--log_interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--knowledge_reuse', type=bool, default=True)
    parser.add_argument('--representation_relation', type=bool, default=True)
    parser.add_argument('--selection_method_for_related_class', type=str, default='mostrelated', 
                        help='Options: mostrelated, leastrelated, random.')
    parser.add_argument('--reuse_from', type=int, default=2, help='start reuse from task')
    parser.add_argument('--l_reuse', type=int, default=3, help='lreuse')
    parser.add_argument('--alloc_prec_conv', type=float, default=0.7)
    parser.add_argument('--alloc_prec_fc', type=float, default=0.2)
    parser.add_argument('--alloc_prec_fc_last_layer', type=float, default=0.1)
    parser.add_argument('--freezed_prec_conv', type=float, default=0.1)
    parser.add_argument('--freezed_prec_fc', type=float, default=0.3)
    parser.add_argument('--freezed_prec_fc_last_layer', type=float, default=1)
    parser.add_argument('--subfree_prec_conv', type=float, default=0.7)
    parser.add_argument('--reuse_prec_conv', type=float, default=0.3)
    parser.add_argument('--subfree_prec_fc', type=float, default=0.7)
    parser.add_argument('--reuse_prec_fc', type=float, default=0.3)
    parser.add_argument('--density_level_conv', type=float, default=0.25)
    parser.add_argument('--density_level_fc', type=float, default=0.25)
    parser.add_argument('--density_level_fc_last_layer', type=float, default=0.7)    
    parser.add_argument('--save_path', type=str, default='./')

    args = parser.parse_args()
    print(args)	

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    # set seed 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create a new directory for saving the results if not exist
    isExist = os.path.exists(args.save_path)
    if not isExist:
        os.makedirs(args.save_path)
        print("The new directory is created!")
	
    ############### Tasks construction ###############
    ## construct the CL tasks with the specified order
    class_order = utils_CNN.construct_class_order(args.class_order, args.benchmark, args.num_tasks, args.num_classes_per_task)
    print(class_order)

    task_labels = []
    for i in range(0, args.num_tasks):
        start_idx = i*args.num_classes_per_task
        end_idx   = i*args.num_classes_per_task + args.num_classes_per_task
        task_labels.append(class_order[start_idx:end_idx]) 
    target_task_labels = []
    for i in range(0, args.num_tasks*args.num_classes_per_task, args.num_classes_per_task):
        target_task_labels.append(list(range(i, i+args.num_classes_per_task))) 
    train_dataset,test_dataset = utils_CNN.task_construction(task_labels, target_task_labels, args.benchmark)
    
    ############### number of neurons for allocations ###############
    feature_maps = [3, 64, 128, 256, 2048, 2048, args.num_tasks*args.num_classes_per_task]
    num_conv = 3 
    enable_reuse = False
    print("enable_reuse_from_task........", args.reuse_from)
    print("l_reuse", args.l_reuse)

    selected_nodes_count, num_freezedNodes_per_layer = utils_CNN.alloc_fix_count_per_layer(args, num_conv, feature_maps)
    sim_sfree_nodes, sim_reused_from_previous = utils_CNN.sFree_reuse_count_per_layer(args, num_conv,
                                                    feature_maps, args.l_reuse, selected_nodes_count)

    print("num_freezedNodes_per_layer", num_freezedNodes_per_layer)
    print("selected_nodes_count", selected_nodes_count)
    print("sfree_nodes",sim_sfree_nodes)
    print("num_reuse",sim_reused_from_previous)

    ############### CL model ###############
    print("density_level_conv", args.density_level_conv)
    print("density_level_fc", args.density_level_fc)
    print("density_level_fc_last_layer", args.density_level_fc_last_layer)
    input_channels = 3
    num_classes = args.num_tasks*args.num_classes_per_task
    model = models_CNN.CNN(input_channels,num_classes, args.density_level_conv, args.density_level_fc,
                args.density_level_fc_last_layer, selected_nodes_count, num_conv, args.l_reuse, args.benchmark).to(device)
    cl = CL.CL(device, num_freezedNodes_per_layer, selected_nodes_count, target_task_labels,model, 
            sim_sfree_nodes, sim_reused_from_previous)
    
    task_labels = target_task_labels

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cl.model.parameters(),lr=args.lr,momentum=args.momentum,weight_decay=args.l2, nesterov=False)

    ############### training ###############
    agnostic_acc_each_task_at_t = []
    task_acc_each_task_at_t = []
    cl.set_init_network_weight()
    for task_idx in range(0,args.num_tasks):   
        ## get data of current task ##
        train_loader = utils_CNN.get_task_load_train(train_dataset[task_idx],args.batch_size)
        val_loader = utils_CNN.get_task_load_train(train_dataset[task_idx],args.batch_size)
        test_loader = utils_CNN.get_task_load_test(test_dataset[task_idx],args.batch_size)
        cl.reset_importance()

        ## train task
        train_acc_along_training_CIL, test_acc_along_training_CIL, train_acc_along_training_TIL, test_acc_along_training_TIL = train_task(args, cl, train_loader, val_loader, test_loader, optimizer, criterion, device, task_idx, task_labels)

        save_statistics_per_task(args, task_idx, test_acc_along_training_CIL, train_acc_along_training_CIL, test_acc_along_training_TIL, train_acc_along_training_TIL)
     
        ## evaluate previous tasks
        cl.set_classifer_to_all_learned_tasks()
        each_task_acc_CIL, average_acc_CIL, each_task_acc_TIL, average_acc_TIL = test_previous_tasks(args, cl.model, criterion, device,task_idx, task_labels, test_dataset)
        agnostic_acc_each_task_at_t.append(each_task_acc_CIL[task_idx])
        task_acc_each_task_at_t.append(each_task_acc_TIL[task_idx])
        forgetting_CIL = calculate_forgetting(each_task_acc_CIL, agnostic_acc_each_task_at_t, task_idx)
        forgetting_TIL = calculate_forgetting(each_task_acc_TIL, task_acc_each_task_at_t, task_idx)

        save_statistics_final_acc(args, each_task_acc_CIL, average_acc_CIL, forgetting_CIL, each_task_acc_TIL, average_acc_TIL, forgetting_TIL, at_middle=True, task_id=task_idx)

        if task_idx<args.num_tasks-1:
            train_loader_val = utils_CNN.get_task_load_train(train_dataset[task_idx+1],args.batch_size)            
            if(args.knowledge_reuse):
                if (task_idx+1)>=args.reuse_from:
                    enable_reuse = True
                else:
                    enable_reuse = False
            # calculate the activation of new classes on model f(t-1)
            if enable_reuse and args.knowledge_reuse and args.representation_relation==True:
                train_loader_val = utils_CNN.get_task_load_train(train_dataset[task_idx+1], args.batch_size)
                t2_F1, t2_F2, t2_F3, t2_F4, t2_L1 = get_layer_reprsentation(cl, model, device, train_loader_val, task_idx+1)
                t2_representations = [t2_F1, t2_F2, t2_F3, t2_F4, t2_L1]
            else:
                t2_representations = None

            cl.prepare_next_task(args.selection_method_for_related_class, enable_reuse, t2_representations)

    # evaluation after learning the whole sequence
    each_task_acc_CIL, average_acc_CIL, each_task_acc_TIL, average_acc_TIL = test_previous_tasks(args, cl.model, criterion, device,task_idx, task_labels, test_dataset)
    forgetting_CIL = calculate_forgetting(each_task_acc_CIL, agnostic_acc_each_task_at_t, args.num_tasks-1)
    forgetting_TIL = calculate_forgetting(each_task_acc_TIL, task_acc_each_task_at_t, args.num_tasks-1)
    print("average_acc_CIL:", average_acc_CIL)
    print("average_acc_TIL:", average_acc_TIL)
    save_statistics_final_acc(args, each_task_acc_CIL, average_acc_CIL, forgetting_CIL, each_task_acc_TIL, average_acc_TIL, forgetting_TIL, at_middle=False, task_id=args.num_tasks)


## test the perfromance on all seen tasks
def test_previous_tasks(args, model, criterion, device, current_task_idx, task_labels, test_dataset):
    total = 0
    total_TIL =0 
    cnt=0
    tasks_acc_CIL = []
    task_acc_TIL = []
    for i in range(current_task_idx+1):
        print('Task '+str(i))
        test_loader=utils_CNN.get_task_load_test(test_dataset[i],args.batch_size)
        val_acc_CIL, val_acc_TIL, cnf_matrix = evaluate(args,i, task_labels, model, criterion, device, test_loader, is_test_set=True)
        print(cnf_matrix)
        tasks_acc_CIL.append(val_acc_CIL)
        task_acc_TIL.append(val_acc_TIL)
        total += val_acc_CIL
        total_TIL+= val_acc_TIL
        cnt +=1
    average_over_tasks = total /cnt
    average_over_tasks_TIL = total_TIL/cnt
    print(f"average acc over {cnt} tasks = {average_over_tasks}")
    print(f"average acc TIL over {cnt} tasks = {average_over_tasks_TIL}")
    return tasks_acc_CIL, average_over_tasks, task_acc_TIL, average_over_tasks_TIL

def print_representation(representation, task_id, name):
    np.savetxt(name + "_task_id"+str(task_id)+".txt",representation.detach().cpu().numpy().flatten())

## evaluate current task
def evaluate(args,current_task, task_labels, model, criterion, device, test_loader, is_test_set=False):
    model.eval()
    test_loss = 0
    correct = 0
    correct_TIL = 0
    n = 0
    y_true = []
    y_pred = []
    y_pred_TIL = []
    with torch.no_grad():
        debug_data_saved = False
        for data, target in test_loader:
            data, target = data.to(device), target.to(device,dtype=torch.int64)
            output,f_x1, f_x2, f_x4, L_x1, L_x2  = model(data)

            multihead=torch.zeros_like(output)
            multihead[:,task_labels[current_task]]=1

            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            y_true += target.tolist()
            y_pred += pred.view_as(target).tolist()
            #print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]

            # task incremental learning
            output = output*multihead
            pred_TIL = output.argmax(dim=1, keepdim=True)
            correct_TIL += pred_TIL.eq(target.view_as(pred_TIL)).sum().item()
            y_pred_TIL += pred_TIL.view_as(target).tolist()

    cnf_matrix = confusion_matrix(y_true, y_pred)
    cnf_matrix_TIL = confusion_matrix(y_true, y_pred_TIL)
    print(cnf_matrix)
    print(cnf_matrix_TIL)
    test_loss /= float(n)
    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    print('\n{}: Average loss: {:.4f}, Accuracy ITL: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct_TIL, n, 100. * correct_TIL / float(n)))
    sys.stdout.flush()                
    return correct / float(n), correct_TIL/float(n), cnf_matrix

# calculate the activation in each layer for new task t using model trained at time step t-1
def get_layer_reprsentation(CL_obj, model, device, test_loader, current_task):
    model.eval()
    num_classes_per_tasks = len(CL_obj.task_labels[current_task])
    average_F1, average_F2, average_F3, average_L1, average_L2 = torch.zeros((num_classes_per_tasks,64)).to(device) , torch.zeros((num_classes_per_tasks,128)).to(device), torch.zeros((num_classes_per_tasks,256)).to(device), torch.zeros((num_classes_per_tasks,2048)).to(device), torch.zeros((num_classes_per_tasks,2048)).to(device)
    total_samples_per_class = torch.zeros((num_classes_per_tasks)).to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device,dtype=torch.int64)
            output, f1 ,f2,f3,l1,l2 = model(data)
            for i in range(num_classes_per_tasks):
                current_class_idxs = (target== CL_obj.task_labels[current_task][i])
                samples_count = (current_class_idxs==True).sum()
                if samples_count>0:
                    average_F1[i]+=f1[current_class_idxs].sum(axis=0).sum(dim=(1, 2))
                    average_F2[i]+=f2[current_class_idxs].sum(axis=0).sum(dim=(1, 2))
                    average_F3[i]+=f3[current_class_idxs].sum(axis=0).sum(dim=(1, 2))
                    average_L1[i]+=l1[current_class_idxs].sum(axis=0)
                    average_L2[i]+=l2[current_class_idxs].sum(axis=0)
                    total_samples_per_class[i]+=samples_count
        for i in range(num_classes_per_tasks):
            average_F1[i]/=total_samples_per_class[i]
            average_F2[i]/=total_samples_per_class[i]
            average_F3[i]/=total_samples_per_class[i]
            average_L1[i]/=total_samples_per_class[i]
            average_L2[i]/=total_samples_per_class[i]
    return  average_F1, average_F2, average_F3, average_L1, average_L2 

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y.cpu()]

def calculate_forgetting(each_task_acc_CIL, agnostic_acc_each_task_at_t, task_idx):
    BWT = 0
    for i in range(task_idx):
        BWT+=(each_task_acc_CIL[i]-agnostic_acc_each_task_at_t[i])
    if task_idx!=0:
        BWT = BWT/(task_idx)
    print("Avg BWT at task_id {} is {}".format(task_idx, BWT))    
    return BWT

def save_statistics_per_task(args, task_idx, test_acc_along_training_CIL, train_acc_along_training_CIL, test_acc_along_training_TIL, train_acc_along_training_TIL):
    file_name_each_task = args.selection_method_for_related_class+"_test_acc_CIL_along_training_seed_"+ str(args.seed)+ "_task_"+ str(task_idx)
    np.savetxt(f"{args.save_path}{file_name_each_task}.txt", test_acc_along_training_CIL)
    file_name_each_task_train = args.selection_method_for_related_class+"_train_acc_CIL_along_training_seed_"+ str(args.seed)+ "_task_"+ str(task_idx)
    np.savetxt(f"{args.save_path}{file_name_each_task_train}.txt", train_acc_along_training_CIL)
    file_name_each_task_TIL = args.selection_method_for_related_class+"_test_acc_TIL_along_training_seed_"+ str(args.seed)+ "_task_"+ str(task_idx)
    np.savetxt(f"{args.save_path}{file_name_each_task_TIL}.txt", test_acc_along_training_TIL)
    file_name_each_task_train_TIL = args.selection_method_for_related_class+"_train_acc_TIL_along_training_seed_"+ str(args.seed)+ "_task_"+ str(task_idx)
    np.savetxt(f"{args.save_path}{file_name_each_task_train_TIL}.txt", train_acc_along_training_TIL)

def save_statistics_final_acc(args, each_task_acc_CIL, average_acc_CIL, forgetting_CIL, each_task_acc_TIL, average_acc_TIL, forgetting_TIL, at_middle, task_id):
    if at_middle == False:
        task_id = ""
    else:
        task_id = "_t_"+str(task_id)
    file_name_each_task = args.selection_method_for_related_class +"_final_acc_CIL_each_task_seed_"+ str(args.seed) + task_id
    file_name_average_acc = args.selection_method_for_related_class + "_average_acc_CIL_seed_" + str(args.seed) + task_id
    file_name_average_bwt = args.selection_method_for_related_class + "_average_bwt_CIL_seed_" + str(args.seed) + task_id
    file_name_each_task_TIL = args.selection_method_for_related_class +"_final_acc_TIL_each_task_seed_"+ str(args.seed) + task_id
    file_name_average_acc_TIL = args.selection_method_for_related_class + "_average_acc_TIL_seed_" + str(args.seed) + task_id
    file_name_average_bwt_TIL = args.selection_method_for_related_class + "_average_bwt_TIL_seed_" + str(args.seed) + task_id
    np.savetxt(f"{args.save_path}{file_name_each_task}.txt", each_task_acc_CIL)
    np.savetxt(f"{args.save_path}{file_name_average_acc}.txt", [average_acc_CIL])
    np.savetxt(f"{args.save_path}{file_name_average_bwt}.txt", [forgetting_CIL])
    np.savetxt(f"{args.save_path}{file_name_each_task_TIL}.txt", each_task_acc_TIL)
    np.savetxt(f"{args.save_path}{file_name_average_acc_TIL}.txt", [average_acc_TIL])
    np.savetxt(f"{args.save_path}{file_name_average_bwt_TIL}.txt", [forgetting_TIL])


if __name__ == '__main__':
   main()