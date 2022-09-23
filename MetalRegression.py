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
# data handling
import numpy
import pandas
import json

# statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# visualizations
import matplotlib.pyplot as plt
import seaborn

# spotify api
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# system data
from os.path import exists
import time
import ast
import math


class MetalRegression:
    _df = ""

    _genres = dict()
    _genre_df = ""

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

        self.append_popularity()

        self.build_genres()
        self.calc_genres()
        self.build_genre_df()

        self.pred_pop_by_genre()
        self.pred_pop_by_energy()
        self.pred_valence_and_danceability_w_pop()
        self.pred_energy_and_acousticness_w_pop()
        self.pred_energy_by_genre()
        self.pred_feats_w_pop()
        self.pred_target_feats_w_pop()

    def build_df(self, csv):

        # load our csv into a temp variable
        temp_df = pandas.read_csv(csv, index_col=0)

        # remove rows that do not have an artist id or songs
        # we remove the one without top songs as this indicates
        # they have no songs uploaded on spotify, this will cause
        # their popularity to be low
        temp_df = temp_df[~pandas.isnull(temp_df["Spotify ID"])]
        temp_df = temp_df[~pandas.isnull(temp_df["Top track features"])]

        # do a deep copy of our now cleaned dataframe
        self._df = temp_df.copy(deep=True).reset_index()

        # delete the temp dataframe to free some memory
        # (sorry for stealing ur job garbage collector)
        del temp_df

    def append_popularity(self):
        popularity = []

        for index, row in self._df.iterrows():
            print(index)
            time.sleep(2)
            artist = self._spotify.artist(artist_id=row["Spotify ID"])
            popularity.append(artist["popularity"])

        self._df["Popularity"] = popularity

    def authorize_spotify(self):
        client_credentials_manager = SpotifyClientCredentials(client_id=self._cid, client_secret=self._scid)
        self._spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    def build_genres(self):
        for index, row in self._df.iterrows():
            for genre in row["Genre"].replace(" (early)", "").replace(" (later)", "").replace(";", ",").strip().split(", "):
                item = genre.lower()
                if item != "metal":
                    item = item.replace("metal", "").strip()
                convert_string = ast.literal_eval(row["Top track features"])
                danceability = []
                energy = []
                key = []
                loudness = []
                mode = []
                speechiness = []
                acousticness = []
                instrumentalness = []
                liveness = []
                valence = []
                tempo = []
                for song in convert_string:
                    danceability.append(song["danceability"])
                    energy.append(song["energy"])
                    key.append(song["key"])
                    loudness.append(song["loudness"])
                    mode.append(song["mode"])
                    speechiness.append(song["speechiness"])
                    acousticness.append(song["acousticness"])
                    instrumentalness.append(song["instrumentalness"])
                    liveness.append(song["liveness"])
                    valence.append(song["valence"])
                    tempo.append(song["tempo"])

                if item in self._genres:
                    self._genres[item]["danceability"] += danceability
                    self._genres[item]["energy"] += energy
                    self._genres[item]["key"] += key
                    self._genres[item]["loudness"] += loudness
                    self._genres[item]["mode"] += mode
                    self._genres[item]["speechiness"] += speechiness
                    self._genres[item]["acousticness"] += acousticness
                    self._genres[item]["instrumentalness"] += instrumentalness
                    self._genres[item]["liveness"] += liveness
                    self._genres[item]["valence"] += valence
                    self._genres[item]["tempo"] += tempo
                    self._genres[item]["popularity"] += [row["Popularity"]]
                else:
                    new = {
                        "danceability": danceability,
                        "energy": energy,
                        "key": key,
                        "loudness": loudness,
                        "mode": mode,
                        "speechiness": speechiness,
                        "acousticness": acousticness,
                        "instrumentalness": instrumentalness,
                        "liveness": liveness,
                        "valence": valence,
                        "tempo": tempo,
                        "popularity": [row["Popularity"]]
                    }

                    self._genres[item] = new

    def calc_genres(self):
        for genre in self._genres:
            self._genres[genre]["popularity_total"] = len(self._genres[genre]["popularity"])
            self._genres[genre]["popularity_mean"] = sum(self._genres[genre]["popularity"]) / self._genres[genre]["popularity_total"]
            self._genres[genre]["popularity_sd"] = math.sqrt(sum([abs(x - self._genres[genre]["popularity_mean"]) for x in self._genres[genre]["popularity"]]) /self._genres[genre]["popularity_total"])
            for each in self._genres[genre].copy():
                if "popularity" not in each:
                    self._genres[genre]["total"] = len(self._genres[genre]["mode"])
                    self._genres[genre][each + "_mean"] = sum(self._genres[genre][each]) / self._genres[genre]["total"]
                    self._genres[genre][each + "_sd"] = math.sqrt(sum([abs(x - self._genres[genre][each + "_mean"]) for x in self._genres[genre][each]]) / self._genres[genre]["total"])

    def build_genre_df(self):
        data = {
            "genre": [],
            "danceability": [],
            "energy": [],
            "key": [],
            "loudness": [],
            "mode": [],
            "speechiness": [],
            "acousticness": [],
            "instrumentalness": [],
            "liveness": [],
            "valence": [],
            "tempo": [],
            "popularity": [],
            "popularity_total": [],
            "popularity_mean": [],
            "popularity_sd": [],
            "total": [],
            "danceability_mean": [],
            "danceability_sd": [],
            "energy_mean": [],
            "energy_sd": [],
            "key_mean": [],
            "key_sd": [],
            "loudness_mean": [],
            "loudness_sd": [],
            "mode_mean": [],
            "mode_sd": [],
            "speechiness_mean": [],
            "speechiness_sd": [],
            "acousticness_mean": [],
            "acousticness_sd": [],
            "instrumentalness_mean": [],
            "instrumentalness_sd": [],
            "liveness_mean": [],
            "liveness_sd": [],
            "valence_mean": [],
            "valence_sd": [],
            "tempo_mean": [],
            "tempo_sd": []
        }

        for genre in self._genres:
            data["genre"].append(genre)
            for each in self._genres[genre]:
                data[each].append(self._genres[genre][each])

        self._genre_df = pandas.DataFrame(data=data)

    def pred_pop_by_genre(self):
        gen = []
        pop = []
        for i, r in self._genre_df.iterrows():
            for p in r["popularity"]:
                gen.append(r["genre"])
                pop.append(p)

        ols_df = pandas.DataFrame(data={"genre": gen, "popularity": pop})
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ genre -1", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_gen_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))


    def pred_pop_by_energy(self):
        energy = []
        pop = []

        for i, r in self._df.iterrows():
            top_track_feat = ast.literal_eval(r["Top track features"])
            for feat in top_track_feat:
                pop.append(r["Popularity"])
                energy.append(feat["energy"])

        ols_df = pandas.DataFrame(data={"energy": energy, "popularity": pop})
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ energy", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_energy_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))

    def pred_valence_and_danceability_w_pop(self):
        valence = []
        danceability = []
        pop = []

        for i, r in self._df.iterrows():
            top_track_feat = ast.literal_eval(r["Top track features"])
            for feat in top_track_feat:
                pop.append(r["Popularity"])
                valence.append(feat["valence"])
                danceability.append(feat["danceability"])

        ols_df = pandas.DataFrame(data={"valence": valence, "danceability": danceability, "popularity": pop})
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ valence + danceability", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_valence_and_danceability_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))

    def pred_energy_and_acousticness_w_pop(self):
        energy = []
        acousticness = []
        pop = []

        for i, r in self._df.iterrows():
            top_track_feat = ast.literal_eval(r["Top track features"])
            for feat in top_track_feat:
                pop.append(r["Popularity"])
                energy.append(feat["energy"])
                acousticness.append(feat["acousticness"])

        ols_df = pandas.DataFrame(data={"energy": energy, "acousticness": acousticness, "popularity": pop})
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ energy + acousticness", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_energy_and_acousticness_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))

    def pred_energy_by_genre(self):
        gen = []
        energy = []
        for i, r in self._genre_df.iterrows():
            for e in r["energy"]:
                gen.append(r["genre"])
                energy.append(e)

        ols_df = pandas.DataFrame(data={"genre": gen, "energy": energy})

        res = smf.ols(formula="energy ~ genre -1", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_gen_w_energy.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))

    def pred_feats_w_pop(self):
        danceability = []
        energy = []
        key = []
        loudness = []
        mode = []
        speechiness = []
        acousticness = []
        instrumentalness = []
        liveness = []
        valence = []
        tempo = []
        pop = []

        for i, r in self._df.iterrows():
            top_track_feat = ast.literal_eval(r["Top track features"])
            for feat in top_track_feat:
                pop.append(r["Popularity"])
                danceability.append(feat["danceability"])
                energy.append(feat["energy"])
                key.append(feat["key"])
                loudness.append(feat["loudness"])
                mode.append(feat["mode"])
                speechiness.append(feat["speechiness"])
                acousticness.append(feat["acousticness"])
                instrumentalness.append(feat["instrumentalness"])
                liveness.append(feat["liveness"])
                valence.append(feat["valence"])
                tempo.append(feat["tempo"])

        data = {
            "danceability": danceability,
            "energy": energy,
            "key": key,
            "loudness": loudness,
            "mode": mode,
            "speechiness": speechiness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "liveness": liveness,
            "valence": valence,
            "tempo": tempo,
            "popularity": pop
        }

        ols_df = pandas.DataFrame(data=data)
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ danceability + energy + key + loudness + mode + speechiness + acousticness + instrumentalness + liveness + valence + tempo", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_feats_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))

    def pred_target_feats_w_pop(self):
        energy = []
        loudness = []
        acousticness = []
        instrumentalness = []
        valence = []
        pop = []

        for i, r in self._df.iterrows():
            top_track_feat = ast.literal_eval(r["Top track features"])
            for feat in top_track_feat:
                pop.append(r["Popularity"])
                energy.append(feat["energy"])
                loudness.append(feat["loudness"])
                acousticness.append(feat["acousticness"])
                instrumentalness.append(feat["instrumentalness"])
                valence.append(feat["valence"])

        data = {
            "energy": energy,
            "loudness": loudness,
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "valence": valence,
            "popularity": pop
        }

        ols_df = pandas.DataFrame(data=data)
        ols_df = ols_df[~(ols_df["popularity"] == 0)]

        res = smf.ols(formula="popularity ~ energy + loudness + acousticness + instrumentalness + valence", data=ols_df).fit()
        p_values = res.summary2().tables[1]['P>|t|']

        with open("pred_target_feats_w_pop.txt", 'w+') as file:
            file.write(str(res.summary()) + "\n")
            with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                file.write("P-Values: \n" + str(p_values))


if __name__ == "__main__":

    # name of our csv from MetalScrapeWrangle
    wrangle_csv = "./compiled_artists_by_R.csv"

    # client and secret client ids for the spotify api
    cid = "0e17ae8ac1be417e847b7e9e57ab7bf4"
    scid = "f1cf289a3f1f40a5868fe2879cffbfe5"

    # initialize stuff
    mr = MetalRegression(csv=wrangle_csv, cid=cid, scid=scid)