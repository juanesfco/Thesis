import openneuro as on

def download(ON_AN,path = '/content/drive/MyDrive/UPRM/MS/Thesis/Programming/'):
  td = path + 'Data/' + ON_AN
  on.download(dataset=ON_AN, target_dir=td)