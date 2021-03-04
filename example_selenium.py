
from selenium import webdriver
from text_stuff import image_contains_text

driver = webdriver.Chrome()
driver.get("https://yandex.ru/")
image = driver.get_screenshot_as_png()
assert image_contains_text(image, "Яндекс")
driver.close()
