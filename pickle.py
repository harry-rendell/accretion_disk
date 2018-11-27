import pickle
import pyPLUTO as pp

#file = open('rho_full', 'rb')
#rho = pickle.load(file)

fileObject = open('/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/rho_full','wb')


wdir = '/Users/iCade/Desktop/CAM/PartIII/PROJECT/python/data/'
D = pp.pload(0,w_dir=wdir)


x1 = D.x1; x2 = D.x2; x3 = D.x3

x = [x1,x2,x3]

pickle.dump(x,fileObject)
fileObject.close()
