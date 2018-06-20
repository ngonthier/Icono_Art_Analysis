import requests
import json
import pymongo
#from sys import maxint
from scipy import ndimage
from io import BytesIO

# Need to do sudo service mongod start before running this file

# Debut avril 2018 il y avait 356170 oeuvres avec images

path_data = '/media/HDD/data/Rijksmuseum/'

num_pages = 101
num_per_page = str(100)
client = pymongo.MongoClient()
db = client.rijks

Drop =False
if Drop:
	db.drop_collection('art')

'''
save_painting

Takes a detail painting JSON dictionary, extracts relevant fields,
and saves the result in the "art" collection of mongo db, if that
work is not already saved.

params
data: painting detail object from API as spec'd at http://rijksmuseum.github.io/

returns None
'''
def save_painting(data):
	if data is None:
		return
		
	paint_id = data.get(u'objectNumber') # string/unique id
	title = data.get(u'title')
	colors_norm = data.get(u'colorsWithNormalization') # array of hex colors; either short array or null
	
	desc = data.get(u'description') # a Dutch description of the painting

	makers = data.get(u'principalMakers') # array of artist objects
	if makers is not None:
		artists = [maker.get(u'name') for maker in makers] # array of artist names
	else:
		artists = []

	plaque_english = data.get(u'plaqueDescriptionEnglish') # always null :(

	dating = data.get(u'dating')
	if dating is not None: 
		date = dating[u'sortingDate'] 
	else:
		date = '0000'

	classification = data.get(u'classification')

	img = data.get(u'webImage') # only metadata, no image data
	url = ''
	if img is not None:
		url = img.get(u'url')   # actual image URL, to pull later

	obj = {
		'obj_id': paint_id,
		'title': title,
		'colors_norm': colors_norm,
		'description': desc,
		'artists': artists,
		'plaque': plaque_english,
		'date': date,
		'classification': classification,
		'url': url
		# 'rgb_array': rgb_array
	}
	try: 
		db.art.insert(obj) 
	except pymongo.errors.DuplicateKeyError: # painting already stored, return
		return

'''
get_painting_json

Iterates indefinitely (until crash?) to collect metadata for all works in the 
rijksmuseum collection. 

'''
def get_painting_json():
	#name = path_data + 'rijks_test.out'
	#outfile = open(name,'wb')
	key = 'JkAwYQtx'
	base = 'https://www.rijksmuseum.nl/api/en/'
	url = base + 'collection'

	numPaintings = 0
	numPaintings_alreadyInDB = 0
	errorImage = 0
	#for type_str in list_type:
		
	for period in range(22):
		period_str = str(period)
		for i in range(num_pages): # collection is returned in pages of length num_per_page
			if i%10 ==0: 
				print(i)
				print('numPaintings',numPaintings)
				print('numPaintings_alreadyInDB',numPaintings_alreadyInDB)
				print('errorImage',errorImage)
			try: # expect r.text to be JSON with 3 keys, which include artObjects (all we need)
				r = requests.get(url + '?key=' + key + '&format=json&imgonly=True&ps=' 
					+ num_per_page + '&p=' + str(i)+'&f.dating.period='+period_str)
				#+'&type='+type_str
				# ps : The number of results per page (between 1 and 100
				# p : The result page. Note that p*ps cannot exceed 10000
			except requests.exceptions.ConnectionError:
				continue

			data = json.loads(r.text)
			paintings = data[u'artObjects'] # actual artworks 
			#print(paintings)
			for painting in paintings:
				numPaintings += 1
				# a late add to skip known paintings (faster than above exception)
				if db.art.count({'obj_id': painting[u'objectNumber']}) != 0:
					numPaintings_alreadyInDB += 1
					continue

				detail_url = url + '/' + painting[u'objectNumber'] + '?key=' + key + '&format=json'
				try: # format: painting details with a lot of fields, above function extracts the relevant ones
					r_detail = requests.get(detail_url)
				except requests.exceptions.ConnectionError:
					print('ConnectionError',painting[u'objectNumber'])
					errorImage +=1
					continue
				painting_data = json.loads(r_detail.text)[u'artObject']
				
				save_painting(painting_data)
		#outfile.write(r.text.encode('utf-8'))



get_painting_json()
#save_painting()

# some shit i wrote to make sure uniqueness was enforced (it was)
obj_ids = set()
urls = set()
num_processed = 0
num_with_url = 0
name_file = path_data + 'painting_urls.tsv'
outfile = open(name_file, 'w')
for painting in db.art.find():
	obj_id = painting['obj_id']
	url = painting['url']
	if url != '':
		outfile.write(obj_id + '\t' + url + '\n')
outfile.close()

# 		num_with_url += 1
# print num_with_url
	# if url in urls or obj_id in obj_ids:
	# 	print obj_id
	# 	print url
	# obj_ids.add(obj_id)
	# urls.add(url)
	# num_processed += 1
	# if num_processed % 1000 == 0:
	# 	print num_processed
