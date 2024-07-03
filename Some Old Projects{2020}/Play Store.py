import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService

WAIT_TIME = 4 
APP_NAME =  name 
NUM_OF_CALL = calls 
USER = os.getlogin() 
FULL_PATH = fr"####"

URL = 'https://play.google.com/store/games'

options = webdriver.ChromeOptions()
options.headless = True 
options.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options = options)
driver.get(URL)


reviews_list = []
ratings_list = []
app_reviews_ratings = {}


def navigate_app():
    print("Scraping data....")
    print("Exercise patience as this may take up to 10 minutes or more.")
    search_icon = driver.find_element(by= By.CLASS_NAME, value="google-material-icons.r9optf")
    search_icon.click()
    search_box = driver.find_element(by= By.CLASS_NAME, value="HWAcU")
    search_box.clear() 
    search_box.send_keys(APP_NAME.lower())
    time.sleep(WAIT_TIME)
    search_box.send_keys(Keys.ENTER) 
    time.sleep(WAIT_TIME)
    search_box.send_keys(Keys.TAB*3, Keys.ENTER)
    open_all_reviews()


def open_all_reviews():
    """This function navigates to the 'See all reviews' link and clicks it"""
    time.sleep(WAIT_TIME)
    buttons = driver.find_elements(by= By.TAG_NAME, value="button")
    buttons[-2].click()
    review_scroll = driver.find_element(by= By.CLASS_NAME, value="VfPpkd-Bz112c-LgbsSe.yHy1rc.eT1oJ.mN1ivc.a8Z62d")
    time.sleep(WAIT_TIME)
    for i in tqdm(range(NUM_OF_CALL)):
        review_scroll.send_keys(Keys.TAB, Keys.END*2)
    collect_reviews()


def collect_reviews():
    time.sleep(WAIT_TIME)
    time.sleep(1)
    reviews = driver.find_elements(by= By.CLASS_NAME, value="h3YV2d") 
    star_ratings = driver.find_elements(by= By.CLASS_NAME, value="iXRFPc")
    time.sleep(WAIT_TIME)
    for (review,rating) in zip(reviews, star_ratings):
        review = review.text
        star_rating = rating.get_attribute("aria-label")
        star_rating = re.findall("\d", star_rating)
        star_rating = star_rating[0] 

        reviews_list.append(review) 
        ratings_list.append(star_rating) 
    app_reviews_ratings["reviews"] = reviews_list
    app_reviews_ratings["ratings"] = ratings_list
    driver.quit()
    save_review_dataframe()


def save_review_dataframe():
    print("Storing data, almost done....")
    reviews_ratings_df = pd.DataFrame(app_reviews_ratings)
    reviews_ratings_df = reviews_ratings_df.iloc[1: ,]
    time.sleep(2)
    reviews_ratings_df.to_csv(FULL_PATH, index=False)
    data_rows = "{:,}".format(reviews_ratings_df.shape[0])
    print("\n"f"{Fore.LIGHTGREEN_EX}{data_rows} rows of data have been saved to downloadas as {APP_NAME.title()}_reviews.csv.")
    print("See you again next time ;-)")


if __name__ == "__main__":
    navigate_app()