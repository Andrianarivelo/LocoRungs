import pdb
import time
import platform
import os
import h5py
import httplib2
import os
from collections import OrderedDict

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage


try:
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

# If modifying these scopes, delete your previously saved credentials
# at ~/.credentials/sheets.googleapis.com-python-quickstart.json
SCOPES = 'https://www.googleapis.com/auth/spreadsheets.readonly'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Google Sheets API Python Quickstart'


class extractData:
    def __init__(self,experiment):
        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' :
            laptop = True
            analysisBase = '/media/HDnyc_data/'
        elif platform.node() == 'otillo':
            laptop = False
            analysisBase = '/media/mgraupe/nyc_data/'
        elif platform.node() == 'bs-analysis':
            laptop = False
            analysisBase = '/home/mgraupe/nyc_data/'
        else:
            print 'Run this script on a server or laptop. Otherwise, adapt directory locations.'
            sys.exit(1)
            
        dataBase     = '/media/invivodata/'
        # check if directory is mounted
        if not os.listdir(dataBase):
            os.system('mount %s' % dataBase)
        if not os.listdir(analysisBase):
            os.system('mount %s' % analysisBase)
        self.dataLocation  = dataBase + 'altair_data/dataMichael/' + experiment + '/'
        self.analysisLocation = analysisBase+'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'
        
        if os.path.exists(self.dataLocation):
            print 'experiment exists'
        else:
            print 'Problem, experiment does not exist'
    
    def getRecordingsList(self):
        recList = [os.path.join(o) for o in os.listdir(self.dataLocation) if os.path.isdir(os.path.join(self.dataLocation,o))]
        #recList = glob(self.dataLocation + '*')
        return recList
        
    def readData(self,recording, device):
        recLocation = self.dataLocation + '/' + recording + '/'
        
        if os.path.exists(self.dataLocation):
            print 'recording exists'
        else:
            print 'Problem, recording does not exist'
            
        if device != 'CameraGigEBehavior':
            pathToFile = recLocation + '%s.ma' % device
        else:
            pathToFile = recLocation + device + '/' + 'frames.ma'
        print pathToFile
        if os.path.isfile(pathToFile):
            fData = h5py.File(pathToFile,'r')
        else:
            print 'file does not exist'
            
        return fData

    def get_credentials(self):
        """Gets valid user credentials from storage.

        If nothing has been stored, or if the stored credentials are invalid,
        the OAuth2 flow is completed to obtain the new credentials.

        Returns:
            Credentials, the obtained credential.
        """
        home_dir = os.path.expanduser('~')
        credential_dir = os.path.join(home_dir, '.credentials')
        if not os.path.exists(credential_dir):
            os.makedirs(credential_dir)
        credential_path = os.path.join(credential_dir,
                                       'sheets.googleapis.com-python-quickstart.json')

        store = Storage(credential_path)
        credentials = store.get()
        if not credentials or credentials.invalid:
            flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
            flow.user_agent = APPLICATION_NAME
            if flags:
                credentials = tools.run_flow(flow, store, flags)
            else:  # Needed only for compatibility with Python 2.6
                credentials = tools.run(flow, store)
            print('Storing credentials to ' + credential_path)
        return credentials
    def convertListToDict(self,values):
        exDict = OrderedDict([])
        for n in range(1,len(values)):
            if values[n][0]:
                print values[n][0]
                # save general information per animal
                exDict[values[n][0]] = OrderedDict([
                                        ('info', {
                                            'type': values[n][1],
                                            'mouse': values[n][2],
                                            'lineage': values[n][3],
                                            'DOB': values[n][4],
                                        },
                                        'dates', {})
                                        ])
                # save days of experiments
                j = n
                while ((not values[j][0]) or (j==n) ) and j < len(values):
                    if values[j][5]:
                        print values[j][5]
                        exDict[values[n][0]]['dates'].update( OrderedDict([
                                                (values[j][5], {
                                                    'age': values[j][6],
                                                    'weight': values[j][7],
                                                    'description': values[j][8],
                                                    'folder' : {},
                                                    'recordings': {}
                                                })
                                                ]))
                        try :
                            test = values[j][10]
                        except IndexError:
                            pass
                        else:
                            exDict[values[n][0]]['dates'][values[j][5]]['folder'] = values[j][10]
                            exDict[values[n][0]]['dates'][values[j][5]]['recordings'].update( OrderedDict([(values[j][11], {
                                                                                                        'comment': values[j][12],} )]))
                    j+=1
        return exDict

    def getExperimentSpreadsheet(self):
        credentials = self.get_credentials()
        http = credentials.authorize(httplib2.Http())
        discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                        'version=v4')
        service = discovery.build('sheets', 'v4', http=http,
                                  discoveryServiceUrl=discoveryUrl)

        # spreadsheetId = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'
        spreadsheetId = '14UbR4oYZLeGchwlGjw_znBwXQUvtaoW7-E-cmDQbr-c'
        rangeName = 'recordings!A:M'
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheetId, range=rangeName).execute()
        values = result.get('values', [])

        exptDict = self.convertListToDict(values)

        return exptDict
        #a = len(values)
        #
        #return values
        #print(a)
        #if not values:
        #    print('No data found.')
        #else:
        #    print('animal ID, data:')
        #    for row in values:
        #        print(len(row))
        #        if row:
        #            # Print columns A and E, which correspond to indices 0 and 4.
        #            print('%s, %s' % (row[0], row[5]))
            
