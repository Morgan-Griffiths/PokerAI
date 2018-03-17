from splinter import Browser

browser = Browser('firefox')
browser.visit('https://www.facebook.com/')
browser.fill('email', 'C5ipo7i@yahoo.com')
browser.fill('pass', 'Darks7o2e')
button = browser.find_by_id('u_0_d')
button.click()
