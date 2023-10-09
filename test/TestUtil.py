import unittest

import torch
import numpy as np
import src.util as util


class MyTestCase(unittest.TestCase):
    def test_label_digit2tensor(self):
        label = [1, 3, 5]
        t = util.label_digit2tensor(label, class_num=10)
        self.assertEqual(t.shape, torch.Size([10]))
        excepted = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 0])
        self.assertTrue(np.all(excepted == t))

    def test_fix_length(self):
        tensor_shape = [1000, 2000, 3000, 4000]
        pad_to = 10000
        for i in tensor_shape:
            print("in i: ")
            t = torch.ones((1, i))
            print(t.shape)
            t = util.fix_length(t, pad_to)
            self.assertEqual(t.shape, torch.Size([1, pad_to]))
            print(t.shape)
            self.assertTrue(
                torch.sum(torch.ones((1, pad_to)) == t) == torch.tensor(i)
            )


if __name__ == '__main__':
    unittest.main()
