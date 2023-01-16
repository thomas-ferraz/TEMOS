import torch
SMPL_JOINTS = {'hips' : 0, 'leftUpLeg' : 1, 'rightUpLeg' : 2, 'spine' : 3, 'leftLeg' : 4, 'rightLeg' : 5,
                'spine1' : 6, 'leftFoot' : 7, 'rightFoot' : 8, 'spine2' : 9, 'leftToeBase' : 10, 'rightToeBase' : 11, 
                'neck' : 12, 'leftShoulder' : 13, 'rightShoulder' : 14, 'head' : 15, 'leftArm' : 16, 'rightArm' : 17,
                'leftForeArm' : 18, 'rightForeArm' : 19, 'leftHand' : 20, 'rightHand' : 21}
# these correspond to [root, left knee, right knee, left heel, right heel, left toe, right toe, left hand, right hand]
# CONTACT_ORDERING = ['hips', 'leftLeg', 'rightLeg', 'leftFoot', 'rightFoot', 'leftToeBase', 'rightToeBase', 'leftHand', 'rightHand']
CONTACT_ORDERING = ['leftFoot', 'rightFoot', 'leftToeBase', 'rightToeBase']
CONTACT_INDS = [SMPL_JOINTS[jname]-1 for jname in CONTACT_ORDERING] #eliminate root node

class ContactLoss:
    def __init__(self):
        self.bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, pred, ground_truth_contact):
        # Contact Loss
        contact_prediction,joints = pred
        velocities = torch.zeros_like(joints)
        velocities[:,:-1,...] = joints[:,1:,...] - joints[:,:-1,...]
        # pred is assumed to be logits from network (i.e. sigmoid has not been applied yet)
        cur_contacts_loss = self.bce_loss(contact_prediction, ground_truth_contact).mean()
        # Vel Loss
        pred_contacts = torch.sigmoid(contact_prediction)
        # use predicted contact probability to weight regularization on joint velocity
        vel_mag = torch.norm(velocities[:,:,CONTACT_INDS,:], dim=-1)
        cur_contact_vel_loss = (pred_contacts*(vel_mag**2)).mean()
        loss = cur_contacts_loss + cur_contact_vel_loss
        return loss

    def __repr__(self):
        return "ContactLoss()"
