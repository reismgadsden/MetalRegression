"""
This program will do a linear regression on data collected
and compiled by MetalScrape.py and MetalWrangle.py.

author: Reis Gadsden 2022-09-16
modified: Reis Gadsden 2022-09-20
class: CS-5245 @ Appalchian State University
instructor: Dr. Mitchell Parry
git: https://github.com/reismgadsden/MetalRegression
"""

"""
IMPORTS
"""
import numpy
import pandas
import statsmodels.api as sm
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from os.path import exists


class MetalRegression:
    _df = ""

    _cid = ""
    _scid = ""
    _spotify = ""

    def __init__(self, csv=None, cid=None, scid=None):

        # if a csv is provided abort
        if csv is None:
            print("You must provide a csv to be loaded.")
            exit(-1)

        # if the csv provided does not exist abort
        elif not exists(csv):
            print("File '" + csv + "' does not exist, please provide a valid file.")
            exit(-1)

        # if client and secret client ids are not provided abort
        elif cid is None or scid is None:
            print("You must provide a client ID and secret ID.")
            exit(-1)

        # load client and secret parameters into corresponding fields
        self._cid = cid
        self._scid = scid

        # build a pandas dataframe from our csv
        self.build_df(csv)

        # authorize a connection to the spotify api
        self.authorize_spotify()

    def build_df(self, csv):
        temp_df = pandas.read_csv(csv, index_col=0)
        temp_df = temp_df[~pandas.isnull(temp_df["Spotify ID"])]
        temp_df = temp_df[~pandas.isnull(temp_df["Top track features"])]
        self._df = temp_df.copy(deep=True)
        del temp_df

    def authorize_spotify(self):
        client_credentials_manager = SpotifyClientCredentials(client_id=self._cid, client_secret=self._scid)
        self._spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)


if __name__ == "__main__":

    # name of our csv from MetalScrapeWrangle
    csv = "./compiled_artists_by_R.csv"

    # client and secret client ids for the spotify api
    cid = ""
    scid = ""

    # initialize stuff
    mr = MetalRegression(csv=csv, cid=cid, scid=scid)