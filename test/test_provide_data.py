import unittest
from unittest.mock import patch
from io import StringIO
import neural.provide_data


class ProvideDataTest(unittest.TestCase):

    @patch('builtins.open')
    @patch('requests.get')
    @patch('neural.provide_data.read_csv_as_bytes')
    @patch('os.listdir')
    @patch('os.mkdir')
    @patch('sys.stdout', new_callable=StringIO)
    def test_no_data_dir_gets_and_loads(self, mock_stdout, mock_mkdir, mock_listdir,
                                        mock_read_csv, mock_get, mock_open):
        neural.provide_data.PATH_DATA = "not_there"
        neural.provide_data.load_train_and_test_data()
        mock_listdir.return_value = tuple()
        # directory does not exist, so mkdir must create it = mock_mkdir is called once
        mock_mkdir.assert_called_once()
        mock_listdir.assert_called_once()
        mock_get.assert_called()
        mock_open.assert_called()
        mock_read_csv.assert_called()
        self.assertEqual('Create Data Directory ...\nDataset fashion-mnist_train.csv loading ... Please wait...\n'
                         + 'Loaded.\nDataset fashion-mnist_test.csv loading ... Please wait...\nLoaded.\n',
                         mock_stdout.getvalue())

    # noinspection PyUnusedLocal
    @patch('requests.get')
    @patch('neural.provide_data.read_csv_as_bytes')
    @patch('os.listdir')
    @patch('os.mkdir')
    def test_data_exists_no_load(self, mock_mkdir, mock_listdir, mock_read_csv, mock_get):
        neural.provide_data.PATH_DATA = "../dataSources"
        mock_listdir.return_value = (neural.provide_data.TRAIN_CSV, neural.provide_data.TEST_CSV)
        neural.provide_data.load_train_and_test_data()
        mock_listdir.assert_called_once()
        mock_read_csv.assert_called()
        mock_get.assert_not_called()
