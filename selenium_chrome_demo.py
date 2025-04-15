from selenium import webdriver
from selenium.webdriver.chrome.options import Options

chrome_options = Options()
chrome_options.debugger_address = "localhost:9222"  # 连接已有 Chrome

driver = webdriver.Chrome(options=chrome_options)
driver.get("https://example.com")
print(driver.title)
