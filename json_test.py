import json

objects = []

for i in range(10):
    object = { "objectLabel":"PowerCell", "x":i*1.1, "y":i*11.0, "z":i*110.0, "confidence":52 }
    objects.append(object)

jsonObjects = json.dumps(objects)

print(jsonObjects)
