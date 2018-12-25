#coding: utf-8

def convert_to_ffm(df,type,numerics,categories,features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}
    # Flagging categorical and numerical fields
    for x in numerics:
         catdict[x] = 0
    for x in categories:
         catdict[x] = 1
    
    nrows = df.shape[0]
    ncolumns = len(features)
    with open(str(type) + "_ffm.txt", "w") as text_file:

    # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['Label']))
            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if(catdict[x]==0):
                    datastring = datastring + " "+str(i)+":"+ str(i)+":"+ str(datarow[x])
                else:
            # For a new field appearing in a training example
                    if(x not in catcodes):
                        catcodes[x] = {}
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
            # For already encoded fields
                    elif(datarow[x] not in catcodes[x]):
                        currentcode +=1
                        catcodes[x][datarow[x]] = currentcode #encoding the feature
                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " "+str(i)+":"+ str(int(code))+":1"

            datastring += '\n'
            text_file.write(datastring)