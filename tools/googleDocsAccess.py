from __future__ import print_function
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
CLIENT_SECRET_FILE = 'tools/client_secret.json'
APPLICATION_NAME = 'Google Sheets API Python Quickstart'


def get_credentials():
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


def convertListToDict(values):
    exDict = OrderedDict([])
    for n in range(1, len(values)):
        if values[n][0]:
            # print values[n][0]
            # save general information per animal
            exDict[values[n][0]] = OrderedDict([
                ('info', {
                    'type': values[n][1],
                    'mouse': values[n][2],
                    'lineage': values[n][3],
                    'DOB': values[n][4],
                }),
                ('dates', {})
            ])
            # save days of experiments
            j = n
            while (j < len(values)) and ((not values[j][0]) or (j == n)):
                if values[j][5]:
                    # print values[j][5]
                    exDict[values[n][0]]['dates'].update(OrderedDict([
                        (values[j][5], {
                            'age': values[j][6],
                            'weight': values[j][7],
                            'description': values[j][8],
                            'folder': {},
                            'recordings': {}
                        })
                    ]))
                    try:
                        values[j][10]
                    except IndexError:
                        pass
                    else:
                        exDict[values[n][0]]['dates'][values[j][5]]['folder'] = values[j][10]
                        m = j
                        # read out recordings
                        while (m<len(values)) and ((not values[m][5]) or (j == m)) :  # values[m][11] :
                            try:
                                values[m][11]
                            except:
                                pass
                            else:
                                exDict[values[n][0]]['dates'][values[j][5]]['recordings'].update(
                                    OrderedDict([(values[m][11], {
                                        'comment': {}, })]))
                                try:
                                    values[m][12]
                                except:
                                    pass
                                else:
                                    exDict[values[n][0]]['dates'][values[j][5]]['recordings'][values[m][11]][
                                        'comment'] = values[m][12]
                            m += 1
                j += 1
    return exDict


def getExperimentSpreadsheet():
    credentials = get_credentials()
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

    exptDict = convertListToDict(values)

    return exptDict
    # a = len(values)
    #
    # return values
    # print(a)
    # if not values:
    #    print('No data found.')
    # else:
    #    print('animal ID, data:')
    #    for row in values:
    #        print(len(row))
    #        if row:
    #            # Print columns A and E, which correspond to indices 0 and 4.
    #            print('%s, %s' % (row[0], row[5]))
