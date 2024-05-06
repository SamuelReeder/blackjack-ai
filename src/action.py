import json
from time import sleep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.action_chains import ActionBuilder
import pytesseract
from PIL import Image

class ActionInterface:
    def __init__(self, data):
        self.load_data(data)
        self.last_clicked_element = None
        self.driver = webdriver.Firefox()
        self.driver.maximize_window()


    def load_data(self, data):
         with open(data, 'r') as f:
            self.data = json.load(f)
        

    def execute(self, name):
        if self.data[name]['url'] != 'none':
            self.driver.get(self.data[name]['url'])
        sleep(3)

        for action in self.data[name]['steps']:
            if action['action'] == 'click':
                element = None
                if action['id'] != 'none':
                    print('ID:', action['id'])
                    element = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.ID, action['id'])))
                # elif action['classes'] != 'none':
                #     print('Classes:', action['classes'])
                #     element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, action['classes'])))
                # elif action['tag']:
                #     print('Tag:', action['tag'])
                #     element = WebDriverWait(driver, 10).until(EC.visibility_of_element_located((By.TAG_NAME, action['tag'])))
                else:
                    # Move cursor to x, y position and click
                    x = action['position']['x']
                    y = action['position']['y']

                    # # Get the last clicked element
                    # Move to position (64,60) and click() at that position (Note: you will not see your mouse move)
                    action = ActionBuilder(self.driver)
                    action.pointer_action.move_to_location(x, y)
                    action.pointer_action.click()
                    action.perform()
                    self.last_clicked_element = self.driver.switch_to.active_element

                if element:
                    element.click()
                    self.last_clicked_element = element

            elif action['action'] == 'keydown' and self.last_clicked_element is not None:
                print('Key:', action['key'])
                self.last_clicked_element.send_keys(action['key'])
                
            sleep(1)
                
    def scan(self):
        # element = self.driver.find_element(By.CSS_SELECTOR, 'body')

        # element.screenshot('element.png')
        self.driver.get_screenshot_as_file('element.png')


        pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update this path

        # Perform OCR on the screenshot
        img = Image.open('element.png')
        text = pytesseract.image_to_string(img)
        print(text)