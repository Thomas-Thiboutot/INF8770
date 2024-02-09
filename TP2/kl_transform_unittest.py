import unittest

import numpy as np

from kl_transform import eqm,psnr


class TestKlTransform(unittest.TestCase):
    
    def test_eqm(self):
        self.assertEqual(eqm(np.array([[[1,1,1]]]), np.array([[[2,2,2]]])), 3, "Should be 3")
        self.assertEqual(eqm(np.array([[[-1,-1,-1]]]), np.array([[[2,2,2]]])), 27, "Should be 27")
        self.assertEqual(eqm(np.array([[[-1,-1,-1], [-1,-1,-1]]]), np.array([[[2,2,2], [2,2,2]]])), 27, "Should be 27")
    
    def test_psnr(self):
        self.assertEqual(round(psnr(1),2) , 48.13, "Should be close to 48.13")
        
if __name__ == '__main__':
    unittest.main()