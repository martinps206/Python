import requests

URL = 'https://httpbin.org/get'
URL_1 = 'https://httpbin.org/get?name=Martin&password=123456&email=martinps_cc@hotmail.com'

######################################
#response = requests.get(URL)  #GET
#print(response.text) #str
#print(type(response.text)) #str
#print(response.status_code)

#payload = response.json()
#print(payload.get('origin'))
#print(response.json()) #dictionary
#print(response.url)
######################################


#query
#response = requests.get(URL_1)  #GET

#if response.status_code == 200:
    #print(response.text)
    #print(response.url)
    #payload = response.json()
    #print(payload.get('args'))

""" params = {
    'email': 'martinps_cc@hotmail.com', 
    'name': 'Martin', 
    'password': '123456'
}

response = requests.get(URL, params=params)

if response.status_code == 200:
    #print(response.text)
    print(response.url)
    payload = response.json()
    print(payload.get('args')) """

