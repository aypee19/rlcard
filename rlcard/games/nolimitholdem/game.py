from enum import Enum

import numpy as np
from copy import deepcopy
from rlcard.games.limitholdem import Game
from rlcard.games.limitholdem import PlayerStatus
from rlcard.games.limitholdem.utils import compare_hands

from rlcard.games.nolimitholdem import Dealer
from rlcard.games.nolimitholdem import Player
from rlcard.games.nolimitholdem import Judger
from rlcard.games.nolimitholdem import Round, Action


class Stage(Enum):

    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    END_HIDDEN = 4
    SHOWDOWN = 5


class NolimitholdemGame(Game):

    def __init__(self, allow_step_back=False, num_players=2):
        ''' Initialize the class nolimitholdem Game
        '''
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()

        # small blind and big blind
        self.small_blind = 1
        self.big_blind = 2 * self.small_blind

        # config players
        self.num_players = num_players
        self.init_chips = [100] * num_players

        # Randomly choose a dealer
        self.dealer_id = self.np_random.randint(0, self.num_players)

        self.record_steps = False

    def configure(self, game_config):
        ''' Specifiy some game specific parameters, such as player number and initial chips
        '''
        self.num_players = game_config['game_player_num']
        self.init_chips = game_config['chips_for_each']
        self.dealer_id = game_config['dealer_id'] if game_config['dealer_id'] is not None else \
            self.np_random.randint(0, self.num_players)
        self.record_steps = game_config['record_steps']

    def init_game(self):
        ''' Initialilze the game of Limit Texas Hold'em

        This version supports two-player limit texas hold'em

        Returns:
            (tuple): Tuple containing:

                (dict): The first state of the game
                (int): Current player's id
        '''
        # Initialize step recorder
        if self.record_steps:
            self.steps_recorder = StepsRecorder(self.num_players)

        # Initilize a dealer that can deal cards
        self.dealer = Dealer(self.np_random)

        # Initilize players to play the game
        self.players = [Player(i, self.init_chips[i], self.np_random) for i in range(self.num_players)]

        # Initialize a judger class which will decide who wins in the end
        self.judger = Judger(self.np_random)

        # Deal cards to each  player to prepare for the first round
        for i in range(2 * self.num_players):
            self.players[i % self.num_players].hand.append(self.dealer.deal_card())

        # Initilize public cards
        self.public_cards = []
        self.stage = Stage.PREFLOP

        # Big blind and small blind
        s = (self.dealer_id + 1) % self.num_players
        b = (self.dealer_id + 2) % self.num_players
        self.players[b].bet(chips=self.big_blind)
        self.players[s].bet(chips=self.small_blind)

        # Record dealer and hands and blinds
        if self.record_steps:
            self.steps_recorder.add_step_dealer_designed(self.dealer_id)
            for i in range(self.num_players):
                player_id = (i + self.dealer_id + 1) % self.num_players
                if self.record_steps:
                    self.steps_recorder.add_step_hand_dealt(player_id,
                                                            [str(card) for card in self.players[player_id].hand])
            self.steps_recorder.add_step_blind(s, self.small_blind)
            self.steps_recorder.add_step_blind(b, self.big_blind)

        # The player next to the small blind plays the first
        self.game_pointer = (b + 1) % self.num_players

        # Initilize a bidding round, in the first round, the big blind and the small blind needs to
        # be passed to the round for processing.
        self.round = Round(self.num_players, self.big_blind, dealer=self.dealer, np_random=self.np_random)

        self.round.start_new_round(game_pointer=self.game_pointer, raised=[p.in_chips for p in self.players])

        # Count the round. There are 4 rounds in each game.
        self.round_counter = 0

        # Save the hisory for stepping back to the last state.
        self.history = []

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_legal_actions(self):
        ''' Return the legal actions for current player

        Returns:
            (list): A list of legal actions
        '''
        return self.round.get_nolimit_legal_actions(players=self.players)

    def step(self, action):
        ''' Get the next state

        Args:
            action (str): a specific action. (call, raise, fold, or check)

        Returns:
            (tuple): Tuple containing:

                (dict): next player's state
                (int): next plater's id
        '''

        if action not in self.get_legal_actions():
            print(action, self.get_legal_actions())
            print(self.get_state(self.game_pointer))
            raise Exception('Action not allowed')

        if self.allow_step_back:
            # First snapshot the current state
            r = deepcopy(self.round)
            b = self.game_pointer
            r_c = self.round_counter
            d = deepcopy(self.dealer)
            p = deepcopy(self.public_cards)
            ps = deepcopy(self.players)
            self.history.append((r, b, r_c, d, p, ps))

        player_acting = self.game_pointer

        # Then we proceed to the next round
        self.game_pointer = self.round.proceed_round(self.players, action)

        if self.record_steps:
            self.steps_recorder.add_step_player_action(player_acting, action, self.players[player_acting].status,
                                                       self.get_state())

        players_in_bypass = [1 if player.status in (PlayerStatus.FOLDED, PlayerStatus.ALLIN) else 0 for player in self.players]
        if self.num_players - sum(players_in_bypass) == 1:
            last_player = players_in_bypass.index(0)
            if self.round.raised[last_player] >= max(self.round.raised):
                # If the last player has put enough chips, he is also bypassed
                players_in_bypass[last_player] = 1

        players_alive = self.num_players - sum(player.status == PlayerStatus.FOLDED for player in self.players)

        # If a round is over, we deal more public cards
        if self.round.is_over() and players_alive > 1:
            # Game pointer goes to the first player not in bypass after the dealer, if there is one
            self.game_pointer = (self.dealer_id + 1) % self.num_players
            if sum(players_in_bypass) < self.num_players:
                while players_in_bypass[self.game_pointer]:
                    self.game_pointer = (self.game_pointer + 1) % self.num_players

            # For the first round, we deal 3 cards
            if self.round_counter == 0:
                self.stage = Stage.FLOP
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                self.public_cards.append(self.dealer.deal_card())
                if self.record_steps:
                    self.steps_recorder.add_step_next_stage(self.stage, [str(card) for card in self.public_cards[-3:]],
                                                            self.get_state())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            # For the following rounds, we deal only 1 card
            if self.round_counter == 1:
                self.stage = Stage.TURN
                self.public_cards.append(self.dealer.deal_card())
                if self.record_steps:
                    self.steps_recorder.add_step_next_stage(self.stage, [str(self.public_cards[-1])],
                                                            self.get_state())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1
            if self.round_counter == 2:
                self.stage = Stage.RIVER
                self.public_cards.append(self.dealer.deal_card())
                if self.record_steps:
                    self.steps_recorder.add_step_next_stage(self.stage, [str(self.public_cards[-1])],
                                                            self.get_state())
                if len(self.players) == np.sum(players_in_bypass):
                    self.round_counter += 1

            self.round_counter += 1
            self.round.start_new_round(self.game_pointer)

        state = self.get_state(self.game_pointer)

        return state, self.game_pointer

    def get_state(self, player_id=None):
        ''' Return state of the game, with player's state if a player_id is given.

        Args:
            player_id (int): player id, can be None.

        Returns:
            (dict): The state of the player
        '''
        self.dealer.pot = np.sum([player.in_chips for player in self.players])

        chips = [self.players[i].in_chips for i in range(self.num_players)]
        legal_actions = self.get_legal_actions()

        if player_id is not None:
            state = self.players[player_id].get_state(self.public_cards, chips, legal_actions)
            if self.record_steps:
                state['steps_record'] = self.steps_recorder.players_record[player_id]
        else:
            state = dict()

        state['stakes'] = [self.players[i].remained_chips for i in range(self.num_players)]
        state['current_player'] = self.game_pointer
        state['pot'] = self.dealer.pot
        state['stage'] = self.stage

        return state

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if len(self.history) > 0:
            self.round, self.game_pointer, self.round_counter, self.dealer, self.public_cards, self.players = self.history.pop()
            return True
        return False

    def get_player_num(self):
        ''' Return the number of players in No Limit Texas Hold'em

        Returns:
            (int): The number of players in the game
        '''
        return self.num_players

    def get_payoffs(self):
        ''' Return the payoffs of the game

        Returns:
            (list): Each entry corresponds to the payoff of one player
        '''
        hands = [p.hand + self.public_cards if p.status in (PlayerStatus.ALIVE, PlayerStatus.ALLIN) else None for p in self.players]
        chips_payoffs = self.judger.judge_game(self.players, hands)

        if self.record_steps:
            self.record_steps_showdown(hands)
            self.steps_recorder.add_step_payoffs_delivered(chips_payoffs, self.get_state())

        return chips_payoffs

    def record_steps_showdown(self, hands):
        ''' Record the steps of the showdown if there is one, especially who shows his hand and in which order

        Args:
            hands (list): Hands of the players, each represented by a list of Cards
        '''

        if sum([hand is not None for hand in hands]) > 1:
            # Players must reveal their cards
            self.steps_recorder.add_step_next_stage(Stage.SHOWDOWN, [], self.get_state())

            steps = self.steps_recorder.full_record

            # Define who is the first player to reveal, by default it is the first after the dealer:
            first_to_reveal = (self.dealer_id + 1) % self.num_players

            # If someone raised on the river, he must be the first to reveal cards
            if steps[-2]['type'] != StepsRecorder.Type.NEXT_STAGE:
                # The river was played
                assert steps[-2]['type'] == StepsRecorder.Type.PLAYER_ACTION
                if steps[-2]['action'] != Action.CHECK:
                    # The river was played and someone raised
                    i = -2
                    # Skip players who folded or called
                    while steps[i]['action'] in [Action.FOLD, Action.CALL]:
                        i -= 1
                    # We necessarily come to a player who raised, he must be the first to reveal his cards
                    assert steps[i]['action'] in [Action.RAISE_POT, Action.RAISE_HALF_POT, Action.ALL_IN]
                    first_to_reveal = steps[i]['player_id']

            hands_ordered_by_reveal = np.roll(hands, -first_to_reveal)
            for i in range(self.num_players):
                if hands_ordered_by_reveal[i] is None:
                    continue

                player_id = (i + first_to_reveal) % self.num_players
                # If the player can win against the adversaries who revealed, he reveals his cards
                if compare_hands([[card.get_index() for card in hand] if hand is not None else None for hand in
                                  hands_ordered_by_reveal[:i + 1]])[-1]:
                    str_hand = [card.get_index() for card in hands[player_id]]
                    self.steps_recorder.add_step_hand_revealed(player_id, str_hand)
                else:
                    self.steps_recorder.add_step_hand_revealed(player_id, None)

        else:
            self.steps_recorder.add_step_next_stage(Stage.END_HIDDEN, [], self.get_state())

    @staticmethod
    def get_action_num():
        ''' Return the number of applicable actions

        Returns:
            (int): The number of actions. There are 6 actions (call, raise_half_pot, raise_pot, all_in, check and fold)
        '''
        return len(Action)


class StepsRecorder():

    class Type(Enum):
        ''' Types of steps in the game '''
        DEALER_DESIGNED = 0
        HAND_DEALT = 1
        BLIND = 2
        PLAYER_ACTION = 3
        NEXT_STAGE = 4
        HAND_REVEALED = 5
        PAYOFFS_DELIVERED = 6

    def __init__(self, num_players):
        self.num_players = num_players
        self.step_id = 0
        self.players_record = [[] for _ in range(num_players)]
        self.full_record = []

    def add_step(self, step, visibility=None):
        if visibility is None:
            visibility = [True] * self.num_players
        step['visibility'] = visibility
        step['id'] = self.step_id
        self.step_id += 1
        self.full_record.append(step)
        for player_record, visible in zip(self.players_record, visibility):
            if visible:
                player_record.append(step)

    def add_step_dealer_designed(self, dealer_id):
        self.add_step(dict(type=self.Type.DEALER_DESIGNED,
                           dealer_id=dealer_id))

    def add_step_hand_dealt(self, player_id, hand):
        self.add_step(dict(type=self.Type.HAND_DEALT,
                           player_id=player_id,
                           hand=hand),
                      [p == player_id for p in range(self.num_players)])

    def add_step_blind(self, player_id, blind):
        self.add_step(dict(type=self.Type.BLIND,
                           player_id=player_id,
                           blind=blind))

    def add_step_player_action(self, player_id, action, player_status, state):
        self.add_step(dict(type=self.Type.PLAYER_ACTION,
                           player_id=player_id,
                           action=action,
                           player_status=player_status,
                           state=state))

    def add_step_next_stage(self, stage, public_cards, state):
        self.add_step(dict(type=self.Type.NEXT_STAGE,
                           stage=stage,
                           public_cards=public_cards,
                           state=state))

    def add_step_hand_revealed(self, player_id, hand):
        self.add_step(dict(type=self.Type.HAND_REVEALED,
                           player_id=player_id,
                           hand=hand))

    def add_step_payoffs_delivered(self, payoffs, state):
        self.add_step(dict(type=self.Type.PAYOFFS_DELIVERED,
                           payoffs=payoffs,
                           state=state))

    @classmethod
    def step_to_str(cls, step):
        return "%s: %s" % (
            step['type'], ', '.join(["%s=%s" % (key, value) for key, value in step.items() if key != 'type']))

#if __name__ == "__main__":
#    game = NolimitholdemGame()
#
#    while True:
#        print('New Game')
#        state, game_pointer = game.init_game()
#        print(game_pointer, state)
#        i = 1
#        while not game.is_over():
#            i += 1
#            legal_actions = game.get_legal_actions()
#            # if i == 3:
#            #     print('Step back')
#            #     print(game.step_back())
#            #     game_pointer = game.get_player_id()
#            #     print(game_pointer)
#            #     legal_actions = game.get_legal_actions()
#
#            action = np.random.choice(legal_actions)
#            # action = input()
#            # if action != 'call' and action != 'fold' and action != 'check':
#            #     action = int(action)
#            print(game_pointer, action, legal_actions)
#            state, game_pointer = game.step(action)
#            print(game_pointer, state)
#
#        print(game.get_payoffs())
