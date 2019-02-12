import math
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss,accuracy_score
from collections import defaultdict

PBP_COLS = ['sets_0','sets_1','games_0','games_1','points_0','points_1','tp_0','tp_1','p0_swp','p0_sp','p1_swp','p1_sp','server']

'''
tracking object for player's year-long performance over time
accepts dates in (year,month)
last_year contains last 12 months stats, most recent to least
'''
class stats_52():
    def __init__(self,date):
        self.most_recent = date
        self.last_year = np.zeros([12,4])

    def time_diff(self,new_date,old_date):
        return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

    def set_month(self,match_date):
        diff = self.time_diff(match_date,self.most_recent)
        if diff>=12:
            self.last_year = np.zeros([12,4])
        elif diff>0:
            self.last_year[diff:] = self.last_year[:12-diff]; self.last_year[:diff] = 0
        self.most_recent = match_date

    def update(self,match_date,match_stats):
        self.set_month(match_date)
        self.last_year[0] = self.last_year[0]+match_stats

'''
tracking object for opponent-adjusted ratings
stores opponent ability at time of match to compare performance against
'''
class adj_stats_52():
    def __init__(self,date):
        self.most_recent = date
        self.last_year = np.zeros([12,6])
        self.adj_sr = [0,0]

    def time_diff(self,new_date,old_date):
        return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

    def set_month(self,match_date):
        diff = self.time_diff(match_date,self.most_recent)
        if diff>=12:
            self.last_year = np.zeros([12,6])
        elif diff>0:
            self.last_year[diff:] = self.last_year[:12-diff]; self.last_year[:diff] = 0
        self.most_recent = match_date
        self.update_adj_sr()

    def update(self,match_date,match_stats):
        self.set_month(match_date)
        self.last_year[0] = self.last_year[0]+match_stats
        self.update_adj_sr()

    # update the player's adjust serve/return ability, based on last twelve months
    def update_adj_sr(self):
        s_pt, r_pt = np.sum(self.last_year[:,1]), np.sum(self.last_year[:,3])
        if s_pt==0 or r_pt==0:
            self.adj_sr = [0,0]
            return
        with np.errstate(divide='ignore', invalid='ignore'):
            f_i = np.sum(self.last_year[:,0])/s_pt
            f_adj = 1 - np.sum(self.last_year[:,4])/s_pt
            g_i = np.sum(self.last_year[:,2])/r_pt
            g_adj = 1 - np.sum(self.last_year[:,5])/r_pt
        self.adj_sr[0] = f_i - f_adj
        self.adj_sr[1] = g_i - g_adj


'''
tracking object for yearly tournament averages
'''
class tny_52():
    def __init__(self,date):
        self.most_recent = date
        self.tny_stats = np.zeros([2,2])
        self.historical_avgs = {}

    def update(self,match_year,match_stats):
        diff = match_year-self.most_recent
        if diff>=2:
            self.tny_stats = np.zeros([2,2])
        elif diff==1:
            self.tny_stats[1] = self.tny_stats[0]; self.tny_stats[0]=0
        self.tny_stats[0] = self.tny_stats[0]+match_stats
        self.most_recent = match_year
        self.historical_avgs[match_year] = (self.tny_stats[0][0],self.tny_stats[0][1])
        return 0 if self.tny_stats[1][1]==0 else self.tny_stats[1][0]/float(self.tny_stats[1][1])

'''
tracking object for common-opponent ratings
stores past year of performance against opponents
'''
class commop_stats_52():
    def __init__(self, date):
        self.last_year = defaultdict(lambda: np.zeros([12, 4]))
        self.most_recent = date;

    def time_diff(self, new_date, old_date):
        return 12*(new_date[0]-old_date[0])+(new_date[1]-old_date[1])

    # TODO: update data for every single opponent, just the one being played (otherwise data )
    def update_player_stats(self, match_date, opponent_name):
        diff = self.time_diff(match_date, self.most_recent)
        if diff>=12:
            self.last_year[opponent_name] = np.zeros([12,4])
        elif diff>0:
            self.last_year[opponent_name][diff:] = self.last_year[opponent_name][:12-diff]
            self.last_year[opponent_name][:diff] = 0

    def update_player_histories(self, match_date, opponent_name):
        for opp_name in np.union1d(opponent_name, self.last_year.keys()):
            self.update_player_stats(match_date, opp_name)

        self.most_recent = match_date

    def update(self, match_date, match_stats, opponent_name):
        self.update_player_histories(match_date, opponent_name)
        self.last_year[opponent_name][0] = self.last_year[opponent_name][0]+match_stats

'''
generate delta between two players relative to shared opponent
delta_i^AB = (spw(A, C_i) - (1 - rpw(A, C_i))) - (spw(B, C_i) - (1 - rpw(B, C_i)))
'''
def generate_delta(p1_stats, p2_stats):
    p1_s_pct, p1_r_pct = p1_stats[0]/float(p1_stats[1]), p1_stats[2]/float(p1_stats[3])
    p2_s_pct, p2_r_pct = p2_stats[0]/float(p2_stats[1]), p2_stats[2]/float(p2_stats[3])
    return (p1_s_pct - (1 - p1_r_pct)) - (p2_s_pct - (1 - p2_r_pct))

'''
return true if total service/return points both greater than zero
'''
def has_stats(last_year_stats):
    return last_year_stats[1] > 0 and last_year_stats[3] > 0

'''
get opponents who have played a match in the past 12 months (more than 0 points)
'''
def get_opponents(player_d, player_name):
    historical_opponents = player_d[player_name].last_year.keys()
    return [opp for opp in historical_opponents if has_stats(np.sum(player_d[player_name].last_year[opp], axis=0))]

'''
compute serve/return parameters, given their common opponent history
'''
def generate_commop_params(player_d, player1, player2):
    p1_opponents, p2_opponents = get_opponents(player_d, player1), get_opponents(player_d, player2)
    common_opponents = np.intersect1d(p1_opponents, p2_opponents)
    if len(common_opponents) == 0:
        return .6, .4

    match_deltas = np.zeros(len(common_opponents))
    for i, comm_op in enumerate(common_opponents):
        p1_match_stats = np.sum(player_d[player1].last_year[comm_op], axis=0)
        p2_match_stats = np.sum(player_d[player2].last_year[comm_op], axis=0)
        comm_op_delta = generate_delta(p1_match_stats, p2_match_stats)
        match_deltas[i] = comm_op_delta
        if np.isnan(comm_op_delta):
            print 'nan here: ', p1_match_stats, p2_match_stats, comm_op

    overall_delta = np.mean(match_deltas)
    if np.isnan(overall_delta):
        print 'nan, match_deltas: ', match_deltas
    return (.6 + overall_delta/2), (.4 + overall_delta/2)

'''
collect 12-month s/r common-opponent performance by player (TODO: get rid of start_ind as input and filter before
passing to this function)
'''
def generate_52_commop_stats(df, start_ind):
    player_d = {}
    start_date = (df['match_year'][start_ind], df['match_month'][start_ind])
    # array w/ 2x1 arrays for each player's 12-month serve/return performance
    match_52_stats = np.zeros([2,len(df), 2])

    w_l = ['w','l']
    for i, row in df.loc[start_ind:].iterrows():
        date = row['match_year'], row['match_month']

        for k, label in enumerate(w_l):
            opponent_name = row[w_l[1-k]+'_name']
            if row[label+'_name'] not in player_d:
                player_d[row[label+'_name']] = commop_stats_52(date)

            # can update player objs before calculating params since players cannot share
            # each other as common opponents
            if row[label+'_swon']==row[label+'_swon'] and row[label+'_svpt']==row[label+'_svpt']:
                match_stats = (row[label+'_swon'],row[label+'_svpt'],row[w_l[1-k]+'_svpt']-\
                                row[w_l[1-k]+'_swon'],row[w_l[1-k]+'_svpt'])
                player_d[row[label+'_name']].update(date, match_stats, opponent_name)

        # can compute common-opponent stats after current match stats inputted
        # (since this won't affect common opponents)
        w_s_pct, w_r_pct = generate_commop_params(player_d, row['w_name'], row['l_name'])

        match_52_stats[0][i] = [w_s_pct, w_r_pct]
        match_52_stats[1][i] = [1 - w_r_pct, 1 - w_s_pct]

    for k,label in enumerate(w_l):
        df[label+'_52_commop_s_pct'] = match_52_stats[k][:,0]
        df[label+'_52_commop_r_pct'] = match_52_stats[k][:,1]

    return df


'''
v3.0 with smarter object construction
use np.array to create arrays from lists; use np.concatenate to combine arrays
figuring out the last three lines here made my function about four times faster...
'''
def enumerate_pbp_V3(s,columns,final_set_extend=0):
    # find the number of S,R,D,A characters and use this to initialize
    # all columns as npy arrays of this length
    length = len(s.replace('.','').replace('/','').replace(';',''))

    sub_matches = ['']; sub_sets = [0]
    t_points,p0_tp, p1_tp = [0,0],[0],[0]
    s_points,p0_sp, p1_sp = [0,0],[0],[0]
    sw_points,p0_swp, p1_swp = [0,0],[0],[0]
    points,p0_p, p1_p = [0,0],[0],[0]
    games,p0_g, p1_g = [0,0],[0],[0]
    sets,p0_s, p1_s = [0,0],[0],[0]
    server = 0; servers=[]

    # divides into s_new, array of games
    s = s.split(';'); s_new = []
    for i in range(len(s)):
        if '.' in s[i]:
            game_str = s[i].split('.')
            game_str[0] += '.'
            s_new += game_str
        else:
            s_new += [s[i]]

    # iterate through each game, a point at a time
    for i in range(len(s_new)-1):
        # update server; up_til_now is everything that has elapsed
        server = (server+1)%2 if i>0 else server
        up_til_now = (';'.join(s_new[:i])+';' if i>0 else ';'.join(s_new[:i])).replace('.;','.')

        # update points and sets if there is a tiebreaker
        if games[0]==games[1]==6 and not (p0_s==p1_s==2 and final_set_extend):
            t_server = server
            mini_games = s_new[i].split('/')
            for j in range(len(mini_games)):
                for l in range(len(mini_games[j])):
                    if mini_games[j][l]!='.':
                        winner = int(mini_games[j][l] in ('S','A') and t_server==1 or mini_games[j][l] in ('R','D') and t_server==0)
                        points[winner]+=1
                        t_points[winner]+=1
                        if winner==server: sw_points[winner]+=1
                        s_points[server]+=1
                        sub_m = up_til_now+'/'.join(mini_games[:j])+'/'+mini_games[j][:l+1] if j>0 else up_til_now+mini_games[j][:l+1]
                        sub_matches.append(sub_m)
                        p0_p.append(points[0]);p1_p.append(points[1])
                        p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                        p0_g.append(games[0]);p1_g.append(games[1])
                        p0_s.append(sets[0]);p1_s.append(sets[1])
                        p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                        p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                        servers.append(t_server)
                    else:
                        sets[winner]+=1
                        points, games = [0,0], [0,0]
                        p0_s[-1],p1_s[-1]=sets[0],sets[1]
                        p0_g[-1],p1_g[-1]=games[0],games[1]
                        p0_p[-1],p1_p[-1]=points[0],points[1]
                        sub_matches[-1] += '.'
                t_server = 1 - t_server

        # otherwise
        else:
            for k in range(len(s_new[i])):
                if s_new[i][k]=='/':
                    print 'ERROR; this should be TIEBREAK'
                    return 0

                if s_new[i][k]!='.':
                    winner = int(s_new[i][k] in ('S','A') and server==1 or s_new[i][k] in ('R','D') and server==0)
                    points[winner]+=1
                    t_points[winner]+=1
                    if winner==server: sw_points[winner]+=1
                    s_points[server]+=1
                    if k==len(s_new[i])-1:
                        sub_matches.append(up_til_now+s_new[i][:k+1]+';')
                        games[winner]+=1
                        points = [0,0]
                    else:
                        sub_matches.append(up_til_now+s_new[i][:k+1])

                    p0_p.append(points[0]);p1_p.append(points[1])
                    p0_tp.append(t_points[0]);p1_tp.append(t_points[1])
                    p0_g.append(games[0]);p1_g.append(games[1])
                    p0_s.append(sets[0]);p1_s.append(sets[1])
                    p0_swp.append(sw_points[0]);p1_swp.append(sw_points[1])
                    p0_sp.append(s_points[0]);p1_sp.append(s_points[1])
                    servers.append(server)

                # backtrack and update previous entries if we finished a set; reset points and games
                elif s_new[i][k]=='.':
                    sets[winner]+=1
                    points, games = [0,0], [0,0]
                    p0_s[-1],p1_s[-1]=sets[0],sets[1]
                    p0_g[-1],p1_g[-1]=games[0],games[1]
                    p0_p[-1],p1_p[-1]=points[0],points[1]
                    sub_matches[-1] += '.'

        points = [0,0]

    columns = np.repeat([columns],[len(servers)],axis=0)
    generated_cols = np.array([p0_s[:-1],p1_s[:-1],p0_g[:-1],p1_g[:-1],p0_p[:-1],p1_p[:-1],p0_tp[:-1],p1_tp[:-1],p0_swp[:-1],p0_sp[:-1],p1_swp[:-1],p1_sp[:-1],servers]).T
    return sub_matches[:-1], np.concatenate([columns,generated_cols],axis=1)

# leave this here so you can modify column names
# NOTE: it is best to keep all the arrays in a list and then concatenate outside the loop
# columns param. specifies which columns to feed into the new dataframe
def generate_df_V2(df_pbp,columns,final_set_extend):
    pbps,dfs = [0]*len(df_pbp),[0]*len(df_pbp)
    for i in xrange(len(df_pbp)):
        info = [df_pbp[col][i] for col in columns]
        a,b = enumerate_pbp_V3(df_pbp['pbp'][i],info,final_set_extend)
        pbps[i],dfs[i] = a, np.asarray(b)

    df = pd.DataFrame(np.concatenate(dfs))
    df.columns = columns + PBP_COLS
    print 'df shape: ', df.shape
    df[df.columns[2:]] = df[df.columns[2:]].astype(float)
    df['score'] = np.concatenate(pbps)
    df['in_lead'] = in_lead(df)
    return df


# optimized function to check who leads, combining boolean indices and functions
def in_lead(df):
    c = np.array(df[['sets_0','games_0','points_0']]) - np.array(df[['sets_1','games_1','points_1']])
    set_d,game_d,point_d = c.T
    leads = np.zeros(len(c))
    set_ind = np.where(set_d!=0)[0]
    game_ind = np.logical_and(set_d==0, game_d!=0).nonzero()[0]
    point_ind = np.logical_and(set_d==0, game_d==0).nonzero()[0]
    leads[set_ind] = set_d[set_ind]>0
    leads[game_ind] = game_d[game_ind]>0
    leads[point_ind] = point_d[point_ind]>0
    return leads

# functions used to parse point-by-point tennis data
def simplify(s):
    s=s.replace('A','S');s=s.replace('D','R')
    sets = s.split('.')
    literal_s=''
    for k in range(len(sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server
        games = sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        # now, iterate through every switch of serve
        for game in games:
            game = game.replace("S",str(server))
            game = game.replace("R",str((server+1)%2))
            literal_s += game
            server =(server+1)%2
    return literal_s

def service_breaks(s):
    ## return the service break advantage ##
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.'); p1_games, p2_games = 0,0
    for k in range(len(all_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server
        games = all_sets[k].split(';');length = len(games)
        next_server = (server+1)%2 if length>12 else (server + len(games))%2
        if k==len(all_sets)-1:
            completed_games = all_sets[k].split(';')[:-1]
            for i in range(len(completed_games)):
                if i!=0:
                    server = (server+1)%2
                game = completed_games[i]
                if server==0 and game[-1]=='S' or server==1 and game[-1]=='R':
                    p1_games += 1
                else:
                    p2_games += 1
        next_server = (server+1)%2 if length > 12 else (server + length)%2
    server = (server+1)%2
    if server==0:
        break_adv = math.ceil(float(p1_games-p2_games)/2)
    else:
        break_adv = math.ceil(float(p1_games-p2_games-1)/2)
    return int(break_adv)

def get_set_order(s):
    s=s.replace('A','S');s=s.replace('D','R')
    # split the string on '.' and count sets up to the second to last entry
    # (if the substring ends on a '.' the last element will be '')
    completed_sets = s.split('.')[:-1]
    sets = ''
    for k in range(len(completed_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server
        games = completed_sets[k].split(';');length = len(games)
        # update length of games (service switches) if there is tiebreak
        if length > 12:
            games = games[:-1] + games[-1].split('/')
            next_server = (server+1)%2
        else:
            next_server = (server + len(games))%2
        final_server = (server + len(games) - 1)%2
        # award set to the player who won the last point of the set
        if final_server==0 and games[-1][-1]=='S':
            sets += '0'
        elif final_server==1 and games[-1][-1]=='R':
            sets += '0'
        else:
            sets += '1'
    return sets

# gets game order of entire match, with sets separated by periods
def get_game_order(s):
    s=s.replace('A','S');s=s.replace('D','R')
    # last entry in this will be '' if we split at the end of a set
    all_sets = s.split('.')[:-1]
    game_s = ''
    for k in range(len(all_sets)):
        server = 0 if k==0 else next_server
        #games = all_sets[k].split(';');length = len(games)
        game_s += get_game_order_sub(all_sets[k] + ';',server) + '.'
        next_server = (server+1)%2 if len(all_sets[k].split(';')) > 12 else (server + len(all_sets[k].split(';')))%2
    return game_s

# takes in s
def get_game_order_sub(s,server):
    games = s.split(';')[:-1]; game_s = ''
    for k in range(len(games)):
        if k==12:
            game_s += str(tbreak_winner(games[k],server))
        else:
            game_s += '0' if server==0 and games[k][-1]=='S' or server==1 and games[k][-1]=='R' else '1'
        server = 1 - server
    return game_s

def tbreak_winner(t_s,server):
    mini_games = t_s.split('/')
    for k in range(1,len(mini_games)):
        server = 1 - server
    return 0 if server==0 and mini_games[-1][-1]=='S' or server==1 and mini_games[-1][-1]=='R' else 1

def predictive_power(col,df):
    # find out how well col does at predicting match winners and losers
    times = 0
    even_indices = []
    for i in range(len(df)):
        if df[col][i][0] > df[col][i][1] and df['winner'][i]==0:
            times += 1
        elif df[col][i][0] < df[col][i][1] and df['winner'][i]==1:
            times += 1
        elif df[col][i][0] == df[col][i][1]:
            even_indices.append(i)
    return times/float(len(df)-len(even_indices)), len(df)-len(even_indices)

# include a pre-match prediction
def do_classify(clf, parameters, indf, featurenames, targetname, target1val, mask, reuse_split=None, score_func=None, n_folds=5, n_jobs=1):
    subdf=indf[featurenames]
    print 'type: ',str(type(clf)).split('.')[-1].split("'")[0]

    Xtrain, Xtest, ytrain, ytest = X[mask], X[~mask], y[mask], y[~mask]
    #print len(Xtrain),len(ytrain)
    clf=clf.fit(Xtrain, ytrain)
    probs_train,probs_test = clf.predict_proba(Xtrain),clf.predict_proba(Xtest)
    train_loss, test_loss = log_loss(ytrain,probs_train,labels=[0,1]),log_loss(ytest,probs_test,labels=[0,1])
    train_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print "############# based on standard predict ################"
    print "Accuracy on training data: %0.2f" % (train_accuracy)
    print "Log Loss on training data: %0.2f" % (train_loss)
    print "Accuracy on test data:     %0.2f" % (test_accuracy)
    print "Log Loss on test data:     %0.2f" % (test_loss)
    return clf, Xtrain, ytrain, Xtest, ytest

def normalize_name(s,tour='atp'):
    if tour=='atp':
        s = s.replace('-',' ')
        s = s.replace('Stanislas','Stan').replace('Stan','Stanislas')
        s = s.replace('Alexandre','Alexander')
        s = s.replace('Federico Delbonis','Federico Del').replace('Federico Del','Federico Delbonis')
        s = s.replace('Mello','Melo')
        s = s.replace('Cedric','Cedrik')
        s = s.replace('Bernakis','Berankis')
        s = s.replace('Hansescu','Hanescu')
        s = s.replace('Teimuraz','Teymuraz')
        s = s.replace('Vikor','Viktor')
        s = s.rstrip()
        s = s.replace('Alex Jr.','Alex Bogomolov')
        s = s.title()
        sep = s.split(' ')
        return ' '.join(sep[:2]) if len(sep)>2 else s
    else:
        return s

def break_point(s):
    s=s.replace('A','S');s=s.replace('D','R')
    all_sets = s.split('.')
    for k in range(len(all_sets)):
        # set server to 0 or 1 at beginning of set, keeping track of all transitions
        server = 0 if k==0 else next_server
        games = all_sets[k].split(';')
        length = len(games)
        next_server = (server+1)%2 if length>12 else (server + length)%2
        if k==len(all_sets)-1:
            last_game = games[-1]
            final_server = (server+len(games[:-1]))%2
            pt_s,pt_r = last_game.count('S'),last_game.count('R')
            b_point = pt_r+1>=4 and pt_r+1>=pt_s
            #print b_point
            #print pt_s,pt_r
            if b_point and final_server:
                return (1,0)
            elif b_point and not final_server:
                return (0,1)
            else:
                return (0,0)

# cols is a list of column sets for logistic regression;
# probs are model-specific probabilities
def validate_results(df,probs,lm_columns,n_splits=5):
    kfold = KFold(n_splits=n_splits,shuffle=True)
    scores = np.zeros([len(lm_columns)+len(probs),2,n_splits]);i=0
    for train_ind,test_ind in kfold.split(df):
        lm = linear_model.LogisticRegression(fit_intercept = True)
        train_df,test_df = df.loc[train_ind],df.loc[test_ind]

        for j,prob_col in enumerate(probs):
            y_preds = test_df[prob_col]>.5
            scores[j][0][i]=accuracy_score(test_df['winner'],y_preds)
            scores[j][1][i]=log_loss(test_df['winner'],test_df[prob_col],labels=[0,1])

        for k,cols in enumerate(lm_columns):
            lm.fit(train_df[cols].values.reshape([len(train_df),len(cols)]),train_df['winner'])
            y_preds = lm.predict(test_df[cols].values.reshape([len(test_df),len(cols)]))
            y_probs = lm.predict_proba(test_df[cols].values.reshape([len(test_df),len(cols)]))
            scores[len(probs)+k][0][i]=accuracy_score(test_df['winner'],y_preds)
            scores[len(probs)+k][1][i]=log_loss(test_df['winner'],y_probs,labels=[0,1])
        i+=1

    for j,prob_col in enumerate(probs):
        print prob_col
        print 'accuracy: ', np.mean(scores[j][0])
        print 'loss: ', np.mean(scores[j][1])

    for i,cols in enumerate(lm_columns):
        print 'lm columns: ',cols
        print 'accuracy: ', np.mean(scores[len(probs)+i][0])
        print 'loss: ', np.mean(scores[len(probs)+i][1])

# test results, given train and test dfs
def test_results(df_train,df_test,probs,lm_columns):
    scores = np.zeros([len(lm_columns)+len(probs),2]);i=0
    lm = linear_model.LogisticRegression(fit_intercept = True)

    for j,prob_col in enumerate(probs):
        y_preds = df_test[prob_col]>.5
        print prob_col
        print 'accuracy: ', accuracy_score(df_test['winner'],y_preds)
        print 'loss: ', log_loss(df_test['winner'],df_test[prob_col],labels=[0,1])

    for k,cols in enumerate(lm_columns):
        lm.fit(df_train[cols].values.reshape([df_train.shape[0],len(cols)]),df_train['winner'])
        y_preds = lm.predict(df_test[cols].values.reshape([df_test.shape[0],len(cols)]))
        y_probs = lm.predict_proba(df_test[cols].values.reshape([df_test.shape[0],len(cols)]))

        print 'lm columns: ', cols
        print 'accuracy: ', accuracy_score(df_test['winner'],y_preds)
        print 'loss: ', log_loss(df_test['winner'],y_probs,labels=[0,1])

        if cols == ['elo_diff_538','sf_elo_diff_538']:
            y_logit_probs = y_probs[:,1]

    return y_logit_probs

def in_dict(x,d):
    return x in d

# function to cross-validate, with no match-overlap between splits (since there are 100-200
# points per match); can use sklearn's GridSearchCV for multiple hyperparameters
def cross_validate(val_df,clf,cols,target,hyper_parameters,n_splits):
    print 'searching for hyperparams...'
    ids = list(set(val_df['match_id']))
    vfunc = np.vectorize(in_dict)
    kfold = KFold(n_splits=n_splits,shuffle=True)
    key = hyper_parameters.keys()[0]
    scores = [[] for k in range(len(hyper_parameters[key]))]

    for train_index,____ in kfold.split(ids):
        train_dict = dict(zip(train_index,[1]*len(train_index)))
        train_ind = vfunc(np.array(val_df['match_id']),train_dict)
        test_ind = (1 - train_ind)==1
        Xtrain, ytrain = val_df[cols][train_ind], np.array(val_df[target][train_ind]).reshape([(sum(train_ind),)])
        Xtest, ytest = val_df[cols][test_ind], np.array(val_df[target][test_ind]).reshape([(sum(test_ind),)])

        # retrieve classification score for every hyper_parameter fed into this function
        # LOOP THROUGH ALL KEYS here if you want to test multiple hyper_params
        for j in xrange(len(hyper_parameters[key])):
            setattr(clf,key,hyper_parameters[key][j])
            clf.fit(Xtrain,ytrain)
            score = clf.score(Xtest,ytest)
            scores[j].append(score)
    for i in range(len(scores)):
        print hyper_parameters[key][i],': ',np.mean(scores[i])
    best_ind = np.argmax([np.mean(a) for a in scores])
    print 'best: ',{key:hyper_parameters[key][best_ind]}
    return {key:hyper_parameters[key][best_ind]}

if __name__=='__main__':
    S = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSSSS.SRRSSRSS;SSSS;RSRSRR;RSRSSS;SSSRS;SSRSS;SSSS;SSSRS;SSSRRRRSR.'
    #S = 'SSSS;RRRR;RSSSS;RSSRRSRSRR;SRRSSS;RRSSRR.SSRRSS;RSSSS;SRRRR;RSRRSR;SRSSRS;SSSRS;SSSS;RRRSSR;'
    S1 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R;'
    S2 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;SSSS;'
    S3 = 'SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSSS;SSRRSRSRSS;SSSRS;RRSSRSSS;SSSRS;S/SS/SR/SS/SS/RS/SS/SS/SS/R.RRRSSR;RSRRR;S'
    S4 = 'SS/R.RRRSSR;RSRRR;SSSS;RSSSS;SSRSS;SRSRSRRSSS;SRSSRS;RRRR;RRSRSS'
    a,b = enumerate_pbp(S,'point')
