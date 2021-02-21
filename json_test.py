import json

class PrettyFloat(float):
    def get(self):
        return round(self, 1)

objects = []

for i in range(10):
    object = { "objectLabel":"PowerCell", "x":round(i*1.1, 1), "y":round(i*11.0, 1), "z":round(i*110.0, 1), "confidence":52 }
    objects.append(object)

jsonObjects = json.dumps(objects, indent=4)

print(jsonObjects)
