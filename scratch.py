def generate_elo(df,counts_i):
    players_list = np.union1d(df.w_name, df.l_name)
    player_count = len(players_list)
    # players_elo = dict(zip(players_list, [elo.Rating() for __ in range(player_count)]))
    players_elo = dict(zip(players_list, [elo.Rating()]*player_count))
    sf_elo = {}
    for sf in ('Hard','Clay','Grass'):
        # sf_elo[sf] = dict(zip(players_list, [elo.Rating() for __ in range(player_count)])) 
        sf_elo[sf] = dict(zip(players_list, [elo.Rating()]*player_count)) 

    elo_1s, elo_2s = [],[]
    sf_elo_1s, sf_elo_2s = [],[]
    elo_obj = elo.Elo_Rater()

    current_month = df['match_month'][0]
    active_players = {current_month: set([])} # active in past twelve months

    # update player elo from every recorded match
    for i, row in df.iterrows():
        if row['match_month'] != current_month:
            current_month = row['match_month']
            active_players[current_month] = set([])

        sf,is_gs = row['surface'],row['is_gs']
        w_name, l_name = row['w_name'], row['l_name']
        w_elo,l_elo = players_elo[w_name],players_elo[l_name]
        elo_1s.append(w_elo.value)
        elo_2s.append(l_elo.value)
        elo_obj.rate_1vs1(w_elo,l_elo,is_gs,counts_i)
        
        if sf in ('Hard','Clay','Grass'):
            w_elo,l_elo = sf_elo[sf][w_name],sf_elo[sf][l_name]
            elo_obj.rate_1vs1(w_elo,l_elo,is_gs,counts_i)

        sf_elo_1s.append(w_elo.value)
        sf_elo_1s.append(l_elo.value)            

    players = active_players.values()
    players = set.union(*players)
    players_elo.values() = [[players_elo[player].value] for player in players]
    for sf in ('Hard','Clay','Grass'):
        for player in players:
            players_elo[player] += [sf_elo[sf][player].value]
    players_elo = pd.DataFrame(players_elo)

    # cols = ['player','elo']
    tag = '_538' if counts_i else ''
    df['w_elo'+tag], df['l_elo'+tag] = elo_1s, elo_2s
    df['w_sf_elo'+tag], df['l_sf_elo'+tag] = sf_elo_1s, sf_elo_2s
    return players_elo, df