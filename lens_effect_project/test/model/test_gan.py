import unittest

from app.model.gan import next_element

class GANTest(unittest.TestCase):
    def next_element_is_one(self):
        m_value = 5

        current = 0
        spectect_current = 1

        current = next_element(current, m_value)

        self.assertEqual(current, spectect_current, 'True')
    
    def next_element_is_zero(self):
        m_value = 5

        current = 5
        spectect_current = 0

        current = next_element(current, m_value)

        self.assertEqual(current, spectect_current, 'True')

    def next_element_is_max(self):
        m_value = 5

        current = 4
        spectect_current = 5

        current = next_element(current, m_value)

        self.assertEqual(current, spectect_current, 'True')
        
if __name__ == '__main__':
    unittest.main()