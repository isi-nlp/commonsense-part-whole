#!/bin/bash
#20-30 minutes each
time zcat wikidata-20180612-truthy-BETA.nt.gz | grep "<http://www.wikidata.org/prop/direct/P361>" > wikidata-partof.csv 
time zcat wikidata-20180612-truthy-BETA.nt.gz | grep "<http://www.wikidata.org/prop/direct/P527>" > wikidata-haspart.csv 
time zcat wikidata-20180612-truthy-BETA.nt.gz | grep "<http://schema.org/name> \"[a-z]*\"@en ." > wikidata-en-lowercase-single-names.csv
