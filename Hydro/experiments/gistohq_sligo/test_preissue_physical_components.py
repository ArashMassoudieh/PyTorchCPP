import unittest
import numpy as np
from preissue_physical_components import box_least_squares,component_design

class ComponentTests(unittest.TestCase):
    def test_recovers_independent_effects(self):
        x=np.array([[1.,0.,1.],[0.,1.,1.],[2.,3.,1.],[3.,1.,1.]])
        c=np.array([.8,.2,.01]);np.testing.assert_allclose(box_least_squares(x,x@c,[1,1,.02]),c,atol=1e-12)
    def test_upper_and_lower_faces(self):
        x=np.eye(3);np.testing.assert_allclose(box_least_squares(x,np.array([3.,0.,.02]),[1,1,.01]),[1,0,.01],atol=1e-12)
    def test_rank_deficient(self):
        x=np.ones((5,2));c=box_least_squares(x,np.ones(5),[1,1]);np.testing.assert_allclose(x@c,np.ones(5),atol=1e-12)
    def test_rejects_bad_domain(self):
        for x in [np.array([[np.nan]]),np.array([[-1.]])]:
            with self.assertRaises(ValueError):box_least_squares(x,np.array([1.]),[1])
        with self.assertRaises(ValueError):box_least_squares(np.eye(2),np.ones(2),[1,0])
    def test_background_and_disabled_storm(self):
        d=np.array([1.,2.,3.]);w=d+np.array([2.,0.,4.]);x=component_design(d,w,True)
        np.testing.assert_array_equal(x@np.array([1.,0.,0.]),d)
        np.testing.assert_array_equal(x@np.array([1.,1.,0.]),w)
    def test_invalid_increment(self):
        with self.assertRaises(ValueError):component_design(np.ones(2),np.zeros(2))
    def test_kkt_conditions(self):
        rng=np.random.default_rng(42)
        for _ in range(20):
            x=rng.uniform(0,2,(20,3));y=rng.uniform(0,3,20);c=box_least_squares(x,y,[1,1,1]);g=x.T@(x@c-y)
            for i in range(3):
                if c[i]<1e-8:self.assertGreaterEqual(g[i],-1e-8)
                elif c[i]>1-1e-8:self.assertLessEqual(g[i],1e-8)
                else:self.assertAlmostEqual(g[i],0,places=8)

if __name__=='__main__':unittest.main()
