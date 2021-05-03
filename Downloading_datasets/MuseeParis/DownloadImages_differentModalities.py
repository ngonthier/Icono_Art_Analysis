#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:30:36 2020

The goal of this script is to download the images of the differents modalities

@author: gonthier
"""

import requests
import json
import pandas as pd


domain_query = """{taxonomyTermQuery(filter: {conditions: [{field: "vid", value: "domaine"}]}) { count entities { entityLabel entityId} } }"""


materiau_query = """{
  taxonomyTermQuery(
    filter: {conditions: [{field: "vid", value: "materiaux_technique"}]}
  ) {
    count
    entities {
      entityLabel
    }
  }
}"""

headers = {
  'Content-Type': 'application/json',
  'auth-token': '85e6240d-05f3-4486-9086-0f590cbf088b',
}


url = 'http://apicollections.parismusees.paris.fr/graphql'
r = requests.post(url, json={"query": materiau_query},headers=headers)
print(r.status_code)
print(r.text)