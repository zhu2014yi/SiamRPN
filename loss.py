import  numpy as np
import  torch
import  torch.nn.functional as F



# def Multiloss(cout,rout,target,lambd):
#     cuda = torch.cuda.is_available()
#     device = torch.device('cuda:0,1' if cuda else 'cpu')
#     """"分类损失函数"""
#     class_pred,class_target=cout,target[:,:,0].long()
#     N=class_target.shape[0]
#     class_target=class_target.reshape(-1,1)
#     #通道转换
#     class_pred=class_pred.reshape(N,10,289)
#     class_pred_0=class_pred[:,::2,:]
#     class_pred_1=class_pred[:,1::2,:]
#     #class_pred=np.array([class_pred_0,class_pred_1])
#     class_pred=torch.cat((class_pred_0,class_pred_1),dim=1)
#     class_pred=class_pred.reshape(N,2,1445)
#     class_pred=class_pred.permute(0,2,1)
#     class_pred=class_pred.reshape(-1,2)
#     pos_index=list(np.where(class_target.cpu()==1)[0])
#     neg_index=list(np.where(class_target.cpu()==0)[0])
#     num_neg_index=len(neg_index)
#     num_pos_index=len(pos_index)
#     total_num=num_pos_index+num_neg_index
#     #assert (num_neg_index+num_pos_index)==N*64,"Error: num is uncorrect!!!"
#     class_pre=class_pred[pos_index+neg_index]
#     class_tag=class_target[pos_index+neg_index]
#     #cuda
#     class_tag=class_tag.squeeze().to(device)
#     class_pre=class_pre.to(device)
#     #neg label 为0 pos label为1
#     weights=torch.tensor([num_pos_index/total_num,num_neg_index/total_num]).to(device)
#     closs=F.cross_entropy(class_pre,class_tag,weight=weights,reduction="none")
#     closs=torch.div(torch.sum(closs),total_num)
#     """回归损失函数"""
#     reg_pred=rout
#     reg_target=target[:,:,1:]
#     # reg_pred=reg_pred.reshape(N,20,289)
#     # reg_pred_0=reg_pred[:,::4,:]
#     # reg_pred_1=reg_pred[:,1::4,:]
#     # reg_pred_2=reg_pred[:,2::4,:]
#     # reg_pred_3=reg_pred[:,3::4,:]
#     # reg_pred=torch.cat((reg_pred_0,reg_pred_1,reg_pred_2,reg_pred_3),dim=1)
#     reg_pred=reg_pred.reshape(N,4,1445)
#     reg_pred=reg_pred.permute(0,2,1)#通道转换
#     reg_pred=reg_pred.reshape(-1,4)
#     reg_target=reg_target.reshape(-1,4)
#
#     reg_pre=reg_pred[pos_index]
#     reg_tag=reg_target[pos_index]
#     reg_pre=reg_pre.to(device)
#     reg_tag=reg_tag.to(device)
#     rloss=F.smooth_l1_loss(reg_pre,reg_tag,reduction="none")
#     rloss=torch.div(torch.sum(rloss),num_pos_index)
#     loss=closs+lambd*rloss
#     return closs,rloss,loss




def Multiloss(cout,rout,target,lambd):
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0,1' if cuda else 'cpu')
    """"分类损失函数"""
    class_pred,class_target=cout,target[:,:,0].long()
    N=class_target.shape[0]
    class_target=class_target.reshape(-1,1)
    #通道转换
    class_pred=class_pred.reshape(N,2,1445)
    class_pred=class_pred.permute(0,2,1)
    class_pred=class_pred.reshape(-1,2)
    pos_index=list(np.where(class_target.cpu()==1)[0])
    neg_index=list(np.where(class_target.cpu()==0)[0])
    num_neg_index=len(neg_index)
    num_pos_index=len(pos_index)
    total_num=num_pos_index+num_neg_index
    #assert (num_neg_index+num_pos_index)==N*64,"Error: num is uncorrect!!!"
    class_pre=class_pred[pos_index+neg_index]
    class_tag=class_target[pos_index+neg_index]
    #cuda
    class_tag=class_tag.squeeze().to(device)
    class_pre=class_pre.to(device)
    #neg label 为0 pos label为1
    weights=torch.tensor([num_pos_index/total_num,num_neg_index/total_num]).to(device)
    closs=F.cross_entropy(class_pre,class_tag,weight=weights,reduction="none")
    closs=torch.div(torch.sum(closs),total_num)
    """回归损失函数"""
    reg_pred=rout
    reg_target=target[:,:,1:]
    reg_pred=reg_pred.reshape(N,4,1445)
    reg_pred=reg_pred.permute(0,2,1)#通道转换
    reg_pred=reg_pred.reshape(-1,4)
    reg_target=reg_target.reshape(-1,4)

    reg_pre=reg_pred[pos_index]
    reg_tag=reg_target[pos_index]
    reg_pre=reg_pre.to(device)
    reg_tag=reg_tag.to(device)
    rloss=F.smooth_l1_loss(reg_pre,reg_tag,reduction="none")
    rloss=torch.div(torch.sum(rloss),num_pos_index)
    loss=closs+lambd*rloss
    return closs,rloss,loss



