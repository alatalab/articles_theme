import json

thesaurus=json.load(open("thesaurus.json"))
print(len(thesaurus.keys()))
def tagged_article(text):
  result={}
  for key in thesaurus.keys():
   # if key in text:
    qualifiers=thesaurus[key]

    if len(qualifiers )>0:

      for qualifier in qualifiers:
        if qualifier in text:
          if key not in result:
            result[key]=0
          result[key]=result[key]+1

      if key in result:
        result[key]=result[key]/len(qualifiers)

  return result

print(tagged_article("Chitosan-based hydrogels are considered as promising biomaterials for tissue engineering. Biological properties of chitosan could be significantly improved by modification of its chemical structure. This study was aimed at characterizing macroporous hydrogels fabricated by freeze-drying technique from chitosan, which has been N-acetylated by 2,2-bis(hydroxymethyl)propionic acid or l,d-lactide. The nature of the acetylated agent was shown to significantly affect hydrogels morphology, swelling behavior, zeta-potential, and protein sorption as well as their degradation by lysozyme. According to scanning electron and confocal laser scanning microscopy, the hydrogels possessed interconnected macroporous network that facilitated cells penetration into the interior regions of the hydrogel. Chemical modification of chitosan significantly influenced L929 cell growth behavior on hydrogel compared to the non-modified chitosan. The proposed chemical strategy for modification of chitosan could be considered as promising approach for improvement of chitosan hydrogels. © 2016 Wiley Periodicals, Inc. J. Appl. Polym. Sci. 2017, 134, 44651. © 2016 Wiley Periodicals, Inc."))