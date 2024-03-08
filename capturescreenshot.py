from selenium import webdriver
from time import *
from datetime import datetime

def captscreehshot(url):
    driver = webdriver.Chrome()
    driver.set_window_size(1400,1050)
    driver.get(url)
    # driver.execute_script("document.getElementById('mrf-popup').style.display = 'none';")
    url_archivo=f"capturas/captura-web-{str(int(datetime.timestamp(datetime.now())))}.png"
    driver.get_screenshot_as_file(url_archivo)
    print("end...")

    return url_archivo