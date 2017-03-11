from xml.etree import ElementTree as ET

parser = ET.iterparse("desc2017.xml")
thesaurus={}
for event, element in parser:
    # element is a whole element
    if element.tag == 'DescriptorRecord':
        descriptorName=element.find('DescriptorName')
        l = descriptorName.getchildren()[0].text

        thesaurus[l] = []


        # print('["'+descriptorName.getchildren()[0].text+'"')
         # do something with this element
         # then clean up
        qualifiers=element.find('AllowableQualifiersList')

        if qualifiers is not None:

          qualifiers = qualifiers.findall('AllowableQualifier')

          for qualifier in qualifiers:
            t = qualifier.find('QualifierReferredTo').find('QualifierName')

            thesaurus[l].append(t.getchildren()[0].text)

        # print(thesaurus[l])
        element.clear()

import json

json.dump(thesaurus,open("thesaurus.json","w"))

