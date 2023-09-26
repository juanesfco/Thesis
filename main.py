import sys
import generateSimulationData as gsd
import bayesianModel as bm
import th_sm

def main(imName, dim, pmin, pmax, qmin, qmax, R, path = 'Data/Simulations/'):
    gsd.main(imName,dim, pmin, pmax, qmin, qmax, R)
    bm.main(pmin, pmax, qmin, qmax, R)
    th_sm.main(imName,dim, pmin, pmax, qmin, qmax, R)

if __name__ == "__main__":
    main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]),int(sys.argv[7]))