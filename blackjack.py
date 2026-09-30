from ultralytics import YOLO
import cv2
import math
import torch

class BlackjackGame:
    def __init__(self, path):
        if torch.cuda.is_available():
            self.model = YOLO(path)
        else:
            self.model = YOLO("models/weights/best_openvino_model/")
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        cv2.namedWindow("Blackjack Tracker", cv2.WINDOW_NORMAL)
        self.seen_cards = set()
        self.hilo_count = 0
        self.reset_hand()

    def reset_hand(self):
        # tasto n, resetta la mano corrente e basta
        self.player_cards = []
        self.dealer_cards = []
        self.player_score = 0
        self.dealer_score = 0
        self.turn = 'PLAYER'

    def reset(self):
        # tasto r, nuova partita e reset di tutto
        self.reset_hand()
        self.seen_cards.clear()
        self.hilo_count = 0

    @staticmethod
    def get_card_value(card_name):
        rank = card_name[:-1]

        if rank in ['J', 'Q', 'K']:
            return 10
        elif rank == 'A':
            return 11
        else:
            return int(rank)
    
    @staticmethod
    def get_hilo_value(card_name):
        rank = card_name[:-1]
        if rank in ['10', 'J', 'Q', 'K', 'A']: return -1
        elif rank in ['7', '8', '9']: return 0
        else: return 1

    def compute_scores(self):
        for hand, is_player in [(self.player_cards, True), (self.dealer_cards, False)]:
            partial_sum = 0
            aces = 0
            for card in hand:
                val = self.get_card_value(card)
                partial_sum += val
                if val == 11: aces += 1
                
            while partial_sum > 21 and aces > 0:
                partial_sum -= 10
                aces -= 1
                
            if is_player: self.player_score = partial_sum
            else: self.dealer_score = partial_sum

    def draw_ui(self, frame):
        font = cv2.FONT_HERSHEY_TRIPLEX

        # in alto a sinistra: PLAYER
        player_color = (0, 255, 0) if self.turn == 'PLAYER' else (150, 150, 150)
        cv2.putText(frame, "PLAYER", (20, 30), font, 1, player_color, 2)
        self.draw_score(frame, self.player_score, self.player_cards, (20, 70), (0, 255, 0))

        # in alto a destra: DEALER
        dealer_color = (255, 0, 0) if self.turn == "DEALER" else (150, 150, 150)
        cv2.putText(frame, "DEALER", (350, 30), font, 1, dealer_color, 2)
        self.draw_score(frame, self.dealer_score, self.dealer_cards, (350, 70), (255, 0, 0))

        # in basso a sx HiLo
        cv2.putText(frame, f"HiLo: {self.hilo_count}", (1800, 1000), font, .7, (255, 153, 255), 1)

        if self.turn == 'GAME OVER':
            self.draw_winner(frame)

    def draw_score(self, frame, score, cards, point, rgb):
        if score > 21:
            cv2.putText(frame, f"BUSTED WITH {score}", point, cv2.FONT_HERSHEY_TRIPLEX, 0.6, rgb, 1)
        elif score == 21 and len(cards) == 2:
            cv2.putText(frame, f"BLACKJACK! {score} (2 CARDS)", point, cv2.FONT_HERSHEY_TRIPLEX, 0.6, rgb, 1)
        else:
            cv2.putText(frame, f"SCORE: {score}", point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, rgb, 1)

    def draw_winner(self, frame):
        if self.player_score <= 21 and (self.dealer_score > 21 or self.player_score > self.dealer_score):
                cv2.putText(frame, "PLAYER WINS!!!", (800, 500), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 2)
        elif self.dealer_score <= 21 and (self.player_score > 21 or self.player_score < self.dealer_score):
            cv2.putText(frame, "DEALER WINS!!!", (800, 500), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 0, 0), 2)
        else:
            cv2.putText(frame, "DRAW", (800, 500), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 2)

    def run(self):
        playing = False

        while True:
            _, frame = self.cap.read()
            if torch.cuda.is_available():
                results = self.model.track(frame, persist=True, conf=.8, imgsz=1024, device=0, verbose=False)
            else:
                results = self.model.track(frame, persist=True, conf=.8, imgsz=640, device='cpu', tracker="bytetrack.yaml", verbose=False)
            boxes = results[0].boxes
    
            if boxes is not None and boxes.id is not None:
                cls = boxes.cls.int().tolist()
                ids = boxes.id.int().tolist()
    
                # xywh = [centro_x, centro_y, larghezza, altezza]
                xywh = boxes.xywh.int().tolist()
                unique_cards_in_frame = []
    
                for i in range(len(ids)):
                    card_name = self.model.names[cls[i]]
                    card_id = ids[i]
                    cx, cy, _, _ = xywh[i]
    
                    duplicated = False
    
                    # Controllo se ho già trovato questa carta nello stesso frame
                    for card_saved in unique_cards_in_frame:
                        _name, _, _cx, _cy = card_saved
                        if card_name == _name:
                            # distanza di Pitagora tra due punti
                            distance = math.hypot(cx - _cx, cy - _cy)
    
                            # se la distanza è minore di un certo numero di pixel allora la carta è la stessa
                            if distance < 300:
                                duplicated = True
                                break
    
                    # se non è duplicato allora lo aggiungo alla lista
                    if not duplicated:
                        unique_cards_in_frame.append((card_name, card_id, cx, cy))
                        if card_name not in self.seen_cards:
                            self.seen_cards.add(card_name)
                            self.hilo_count += self.get_hilo_value(card_name)
                            if self.turn == 'PLAYER':
                                self.player_cards.append(card_name)
                            else:
                                self.dealer_cards.append(card_name)

                self.compute_scores()
    
            annotated_frame = results[0].plot()
            if playing == True:
                self.draw_ui(annotated_frame)
            cv2.imshow("Blackjack Tracker", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                self.reset()
            if key == ord('n'):
                self.reset_hand()
            if key == ord('s'):
                if self.turn == 'PLAYER':
                    self.turn = 'DEALER'
                elif self.turn == 'DEALER':
                    self.turn = 'GAME OVER'
            if key == ord('p'):
                playing = not playing

        self.cap.release()
        cv2.destroyAllWindows()