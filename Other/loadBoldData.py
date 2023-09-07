import nibabel as nib

def load(ON_AN,exp_name,subject,run,path='/content/drive/MyDrive/UPRM/MS/Thesis/Programming/'):
  data1 = path + 'Data/' + ON_AN + '/sub-' + subject + '/func/sub-' + subject + '_task-' + exp_name + '_run-' + run + '_bold.nii.gz'
  return(nib.load(data1))
