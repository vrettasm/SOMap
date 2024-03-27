import unittest

import numpy as np
from src.som import SOM


class TestSOM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        print(" >> TestSOM - START -")
    # _end_def_

    @classmethod
    def tearDownClass(cls) -> None:
        print(" >> TestSOM - FINISH -")
    # _end_def_

    def setUp(self) -> None:
        """
        Creates the test object with default setting.

        :return: None.
        """

        # Test object with default settings.
        self.test_som = SOM()
    # _end_def_

    def test_out_of_bounds(self) -> None:
        """
        Test the out_of_bounds(upper, idx) function.
        The valid functionality is [0 <= idx <= upper].

        :return: None.
        """
        idx = 15
        self.assertTrue(self.test_som.out_of_bounds(10, idx))
    # _end_def_

    def test_init(self) -> None:
        """
        Test the __init__() function.

        :return: None.
        """

        with self.assertRaises(ValueError):
            # The grid size should be positive.
            _ = SOM(m=-1)
        # _end_with_

        with self.assertRaises(ValueError):
            # The input dimension should be positive.
            _ = SOM(d=-1)
        # _end_with_

        with self.assertRaises(ValueError):
            # The u_range should be (low_lim, high_lim).
            _ = SOM(u_range=(1, -1))
        # _end_with_

        with self.assertRaises(ValueError):
            # Supported metrics are: all the scipy.spatial.distances
            # with default parameters.
            _ = SOM(metric="TBD")
        # _end_with_

    # _end_def_

    def test_predict(self) -> None:
        """
        Test the predict() function.

        :return: None.
        """

        with self.assertRaises(RuntimeError):
            # Random test dataset.
            x_data = np.random.randn(10, 3)

            # The predict method works only if the model is fitted.
            self.test_som.predict(x_data)
        # _end_with_

    # _end_def_

    def test_reset_network(self) -> None:
        """
        After resetting the network the trained flag should be false.

        :return: None.
        """

        # Set the trained flat to True.
        self.test_som._is_trained = True

        # Reset the network.
        self.test_som.reset_network()

        # Here the flag should be reset to False.
        self.assertFalse(self.test_som._is_trained)

    # _end_def_

# _end_class_


if __name__ == '__main__':
    unittest.main()
