import json

class IDSample:
    def __init__(self, id=None, name=None, address=None, dob=None, national_id=None,
                 gender=None, end_date=None, profession=None, martial_status=None,religion=None):
        self.Id = id
        self.Name = name
        self.Address = address
        self.DOB = dob
        self.NationalID = national_id
        self.Gender = gender
        self.EndDate = end_date
        self.Profession = profession
        self.MartialStat = martial_status
        self.religion = religion

def convert_to_json(sample):
    return json.dumps(sample.__dict__)

