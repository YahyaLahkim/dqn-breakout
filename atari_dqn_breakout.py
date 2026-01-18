"""
Deep Q-Network (DQN) pour Atari Breakout
Implémentation complète avec entraînement et évaluation
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Configuration des paramètres
class Config:
    # Environnement
    ENV_NAME = "ALE/Breakout-v5"
    RENDER_MODE = None  # None pour l'entraînement, "rgb_array" pour visualisation
    
    # Hyperparamètres DQN
    GAMMA = 0.99                    # Facteur de discount
    LEARNING_RATE = 0.00025         # Taux d'apprentissage
    MEMORY_SIZE = 100000            # Taille du replay buffer
    BATCH_SIZE = 32                 # Taille du batch
    EPSILON_START = 1.0             # Exploration initiale
    EPSILON_MIN = 0.1               # Exploration minimale
    EPSILON_DECAY = 0.995           # Décroissance de epsilon
    TARGET_UPDATE = 1000            # Fréquence de mise à jour du réseau cible
    
    # Entraînement
    NUM_EPISODES = 500              # Nombre d'épisodes d'entraînement
    MAX_STEPS = 10000               # Steps max par épisode
    LEARNING_START = 10000          # Steps avant de commencer l'apprentissage
    
    # Prétraitement des frames
    FRAME_STACK = 4                 # Nombre de frames empilées
    FRAME_RESIZE = (84, 84)         # Taille de redimensionnement
    
    # Sauvegarde
    SAVE_DIR = "C:\\Users\\poste\\Desktop\\SIME\\S1\\RL\\Projet Deep Learning\\Modele\\projet"
    MODEL_PATH = os.path.join(SAVE_DIR, "dqn_breakout_model.pth")
    CHECKPOINT_INTERVAL = 50        # Sauvegarder tous les N épisodes


class DQNetwork(nn.Module):
    """
    Réseau de neurones convolutif pour DQN
    Architecture inspirée de l'article original de DeepMind (2015)
    """
    def __init__(self, input_shape, num_actions):
        super(DQNetwork, self).__init__()
        
        # Couches convolutives pour extraire les features des images
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculer la taille après les convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Couches fully connected
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
    def _get_conv_output_size(self, shape):
        """Calcule la taille de sortie des couches convolutives"""
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """Forward pass"""
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ReplayMemory:
    """
    Experience Replay Buffer
    Stocke les transitions (état, action, récompense, état suivant, done)
    """
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Ajoute une transition à la mémoire"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Échantillonne un batch aléatoire"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class FramePreprocessor:
    """
    Prétraitement des frames Atari
    - Conversion en niveaux de gris
    - Redimensionnement
    - Normalisation
    """
    def __init__(self, resize_shape=(84, 84)):
        self.resize_shape = resize_shape
    
    def preprocess(self, frame):
        """Prétraite une frame"""
        # Convertir en niveaux de gris
        gray = np.mean(frame, axis=2).astype(np.uint8)
        
        # Redimensionner
        from scipy.ndimage import zoom
        scale = (self.resize_shape[0] / gray.shape[0], 
                 self.resize_shape[1] / gray.shape[1])
        resized = zoom(gray, scale, order=1)
        
        # Normaliser entre 0 et 1
        normalized = resized / 255.0
        
        return normalized.astype(np.float32)


class DQNAgent:
    """
    Agent DQN pour jouer à Atari Breakout
    """
    def __init__(self, state_shape, num_actions, config):
        self.config = config
        self.num_actions = num_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du device: {self.device}")
        
        # Réseaux (policy et target)
        self.policy_net = DQNetwork(state_shape, num_actions).to(self.device)
        self.target_net = DQNetwork(state_shape, num_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimiseur
        self.optimizer = optim.Adam(self.policy_net.parameters(), 
                                     lr=config.LEARNING_RATE)
        
        # Replay memory
        self.memory = ReplayMemory(config.MEMORY_SIZE)
        
        # Epsilon pour exploration
        self.epsilon = config.EPSILON_START
        
        # Compteurs
        self.steps_done = 0
        
    def select_action(self, state, training=True):
        """Sélectionne une action selon epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def optimize_model(self):
        """Effectue une étape d'optimisation"""
        if len(self.memory) < self.config.BATCH_SIZE:
            return None
        
        # Échantillonner un batch
        transitions = self.memory.sample(self.config.BATCH_SIZE)
        batch = list(zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(batch[1]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Calculer Q(s,a) actuel
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Calculer Q(s',a') pour le target
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.config.GAMMA * next_q_values
        
        # Calculer la perte (Huber loss)
        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
        
        # Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Met à jour le réseau cible"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Décroissance de epsilon"""
        self.epsilon = max(self.config.EPSILON_MIN, 
                          self.epsilon * self.config.EPSILON_DECAY)
    
    def save(self, path):
        """Sauvegarde le modèle"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)
        print(f"Modèle sauvegardé: {path}")
    
    def load(self, path):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        print(f"Modèle chargé: {path}")


class FrameStackEnv:
    """
    Wrapper pour empiler plusieurs frames consécutives
    Permet au réseau d'avoir une information sur la vitesse et direction
    """
    def __init__(self, env, num_stack=4):
        self.env = env
        self.num_stack = num_stack
        self.preprocessor = FramePreprocessor()
        self.frames = deque(maxlen=num_stack)
    
    def reset(self):
        obs, info = self.env.reset()
        frame = self.preprocessor.preprocess(obs)
        for _ in range(self.num_stack):
            self.frames.append(frame)
        return self._get_state(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.preprocessor.preprocess(obs)
        self.frames.append(frame)
        return self._get_state(), reward, terminated, truncated, info
    
    def _get_state(self):
        return np.array(self.frames)
    
    def close(self):
        self.env.close()


def train_dqn(config=Config()):
    """
    Fonction principale d'entraînement
    """
    # Créer l'environnement
    env = gym.make(config.ENV_NAME, render_mode=config.RENDER_MODE)
    env = FrameStackEnv(env, num_stack=config.FRAME_STACK)
    
    # Initialiser l'agent
    state_shape = (config.FRAME_STACK, 84, 84)
    num_actions = env.env.action_space.n
    agent = DQNAgent(state_shape, num_actions, config)
    
    # Statistiques
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT DQN - ATARI BREAKOUT")
    print(f"{'='*60}")
    print(f"Nombre d'épisodes: {config.NUM_EPISODES}")
    print(f"Device: {agent.device}")
    print(f"Architecture: {sum(p.numel() for p in agent.policy_net.parameters())} paramètres")
    print(f"{'='*60}\n")
    
    # Boucle d'entraînement
    for episode in range(config.NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = []
        
        for step in range(config.MAX_STEPS):
            # Sélectionner une action
            action = agent.select_action(state, training=True)
            
            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Clip la récompense entre -1 et 1 (technique standard pour Atari)
            clipped_reward = np.clip(reward, -1, 1)
            
            # Stocker la transition
            agent.memory.push(state, action, clipped_reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            agent.steps_done += 1
            
            # Optimiser le modèle
            if agent.steps_done >= config.LEARNING_START:
                loss = agent.optimize_model()
                if loss is not None:
                    episode_loss.append(loss)
                
                # Mettre à jour le réseau cible
                if agent.steps_done % config.TARGET_UPDATE == 0:
                    agent.update_target_network()
            
            if done:
                break
        
        # Décroissance de epsilon
        agent.decay_epsilon()
        
        # Enregistrer les statistiques
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Affichage
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_length = np.mean(episode_lengths[-10:])
            print(f"Épisode {episode + 1}/{config.NUM_EPISODES} | "
                  f"Récompense: {episode_reward:.1f} | "
                  f"Moyenne (10): {avg_reward:.1f} | "
                  f"Steps: {episode_length} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {len(agent.memory)}")
        
        # Sauvegarde périodique
        if (episode + 1) % config.CHECKPOINT_INTERVAL == 0:
            agent.save(config.MODEL_PATH)
    
    # Sauvegarde finale
    agent.save(config.MODEL_PATH)
    
    # Créer et sauvegarder les graphiques
    plot_training_results(episode_rewards, episode_lengths, losses, config)
    
    env.close()
    return agent, episode_rewards


def plot_training_results(rewards, lengths, losses, config):
    """
    Crée des graphiques des résultats d'entraînement
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Récompenses
    axes[0].plot(rewards, alpha=0.6, label='Récompense par épisode')
    if len(rewards) >= 10:
        moving_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[0].plot(range(9, len(rewards)), moving_avg, 'r-', linewidth=2, 
                     label='Moyenne mobile (10 épisodes)')
    axes[0].set_xlabel('Épisode')
    axes[0].set_ylabel('Récompense totale')
    axes[0].set_title('Évolution des récompenses pendant l\'entraînement')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Longueur des épisodes
    axes[1].plot(lengths, alpha=0.6, label='Durée par épisode')
    if len(lengths) >= 10:
        moving_avg = np.convolve(lengths, np.ones(10)/10, mode='valid')
        axes[1].plot(range(9, len(lengths)), moving_avg, 'g-', linewidth=2,
                     label='Moyenne mobile (10 épisodes)')
    axes[1].set_xlabel('Épisode')
    axes[1].set_ylabel('Nombre de steps')
    axes[1].set_title('Durée des épisodes')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Perte
    if losses:
        axes[2].plot(losses, alpha=0.6, label='Loss')
        if len(losses) >= 10:
            moving_avg = np.convolve(losses, np.ones(10)/10, mode='valid')
            axes[2].plot(range(9, len(losses)), moving_avg, 'orange', linewidth=2,
                         label='Moyenne mobile (10 épisodes)')
        axes[2].set_xlabel('Épisode')
        axes[2].set_ylabel('Loss')
        axes[2].set_title('Évolution de la fonction de perte')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(config.SAVE_DIR, 'training_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nGraphiques sauvegardés: {plot_path}")
    plt.close()


def evaluate_agent(model_path, num_episodes=5, render=True, save_video=False, display_live=True):
    """
    Évalue un agent entraîné avec affichage en temps réel ou création de vidéo
    
    Args:
        model_path: Chemin vers le modèle entraîné
        num_episodes: Nombre d'épisodes d'évaluation
        render: Active le rendu (obsolète, utilisez display_live)
        save_video: Si True, sauvegarde une vidéo au lieu d'afficher
        display_live: Si True, affiche le jeu en temps réel (mode interactif)
    """
    config = Config()
    
    # Choisir le mode de rendu
    if display_live and not save_video:
        # Mode affichage en temps réel
        render_mode = "human"
        print("\n🎮 Mode affichage en temps réel activé")
        print("   Une fenêtre va s'ouvrir pour montrer le jeu\n")
    elif save_video:
        # Mode enregistrement vidéo
        render_mode = "rgb_array"
        print("\n📹 Mode enregistrement vidéo activé\n")
    else:
        # Pas de rendu
        render_mode = None
        print("\n📊 Mode évaluation sans rendu\n")
    
    env = gym.make(config.ENV_NAME, render_mode=render_mode)
    env = FrameStackEnv(env, num_stack=config.FRAME_STACK)
    
    # Charger l'agent
    state_shape = (config.FRAME_STACK, 84, 84)
    num_actions = env.env.action_space.n
    agent = DQNAgent(state_shape, num_actions, config)
    agent.load(model_path)
    agent.epsilon = 0.05  # Faible exploration pour l'évaluation
    
    rewards = []
    frames_collection = []
    
    print(f"{'='*60}")
    print(f"ÉVALUATION DE L'AGENT")
    print(f"{'='*60}\n")
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_frames = []
        step_count = 0
        
        print(f"🎮 Épisode {episode + 1}/{num_episodes} en cours...")
        
        while True:
            # Enregistrer les frames si nécessaire (pour vidéo)
            if save_video:
                frame = env.env.render()
                episode_frames.append(frame)
            
            # L'agent choisit une action
            action = agent.select_action(state, training=False)
            
            # Exécuter l'action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # En mode display_live, le rendu est automatique via render_mode="human"
            # Pas besoin d'appeler env.render() manuellement
            
            if done:
                break
        
        rewards.append(episode_reward)
        
        # Sauvegarder les frames du premier épisode pour la vidéo
        if save_video and episode == 0:
            frames_collection = episode_frames
        
        print(f"   ✓ Épisode {episode + 1}: Récompense = {episode_reward:.1f}, Steps = {step_count}\n")
    
    avg_reward = np.mean(rewards)
    print(f"{'='*60}")
    print(f"📊 RÉSULTATS")
    print(f"{'='*60}")
    print(f"Récompense moyenne: {avg_reward:.2f}")
    print(f"Écart-type: {np.std(rewards):.2f}")
    print(f"Récompense min: {np.min(rewards):.1f}")
    print(f"Récompense max: {np.max(rewards):.1f}")
    print(f"{'='*60}\n")
    
    # Créer la vidéo si demandé
    if save_video and frames_collection:
        save_video_from_frames(frames_collection, config)
    
    env.close()
    return rewards


def save_video_from_frames(frames, config, fps=30):
    """
    Sauvegarde une vidéo à partir d'une liste de frames
    """
    try:
        import cv2
        
        video_path = os.path.join(config.SAVE_DIR, 'breakout_gameplay.mp4')
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Convertir RGB en BGR pour OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        print(f"Vidéo sauvegardée: {video_path}")
        
    except ImportError:
        print("OpenCV non disponible. Installation avec: pip install opencv-python")


def play_live(model_path, num_episodes=3, fps_limit=30):
    """
    Joue en direct avec affichage en temps réel
    Fonction simplifiée pour démonstration interactive
    
    Args:
        model_path: Chemin vers le modèle entraîné
        num_episodes: Nombre d'épisodes à jouer
        fps_limit: Limite de FPS pour ralentir l'affichage (None = vitesse max)
    """
    import time
    
    config = Config()
    
    # Mode human pour affichage direct
    env = gym.make(config.ENV_NAME, render_mode="human")
    env = FrameStackEnv(env, num_stack=config.FRAME_STACK)
    
    # Charger l'agent
    state_shape = (config.FRAME_STACK, 84, 84)
    num_actions = env.env.action_space.n
    agent = DQNAgent(state_shape, num_actions, config)
    agent.load(model_path)
    agent.epsilon = 0.05
    
    print(f"\n{'='*60}")
    print(f"🎮 MODE JEU EN DIRECT")
    print(f"{'='*60}")
    print(f"Episodes: {num_episodes}")
    print(f"FPS limite: {fps_limit if fps_limit else 'Aucune (vitesse max)'}")
    print(f"{'='*60}\n")
    print("Une fenêtre va s'ouvrir...")
    print("Appuyez sur Ctrl+C pour arrêter\n")
    
    frame_time = 1.0 / fps_limit if fps_limit else 0
    
    try:
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            step_count = 0
            
            print(f"🎮 Épisode {episode + 1}/{num_episodes}")
            
            while True:
                start_time = time.time()
                
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                episode_reward += reward
                step_count += 1
                
                # Limiter les FPS si nécessaire
                if fps_limit:
                    elapsed = time.time() - start_time
                    sleep_time = frame_time - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                if done:
                    break
            
            print(f"   ✓ Score: {episode_reward:.1f} | Steps: {step_count}\n")
    
    except KeyboardInterrupt:
        print("\n⚠️  Arrêt demandé par l'utilisateur")
    
    finally:
        env.close()
        print("\n✅ Fenêtre fermée")


def create_performance_report(rewards_train, rewards_eval, config):
    """
    Crée un rapport de performance détaillé
    """
    report = f"""
{'='*70}
RAPPORT DE PERFORMANCE - DQN SUR ATARI BREAKOUT
{'='*70}

1. CONFIGURATION DE L'ENTRAÎNEMENT
   - Algorithme: Deep Q-Network (DQN)
   - Environnement: {config.ENV_NAME}
   - Nombre d'épisodes: {config.NUM_EPISODES}
   - Taille du replay buffer: {config.MEMORY_SIZE}
   - Batch size: {config.BATCH_SIZE}
   - Learning rate: {config.LEARNING_RATE}
   - Gamma (discount factor): {config.GAMMA}

2. ARCHITECTURE DU RÉSEAU
   - Input: {config.FRAME_STACK} frames empilées de {config.FRAME_RESIZE[0]}x{config.FRAME_RESIZE[1]} pixels
   - Couche Conv1: 32 filtres, kernel 8x8, stride 4
   - Couche Conv2: 64 filtres, kernel 4x4, stride 2
   - Couche Conv3: 64 filtres, kernel 3x3, stride 1
   - Couche FC1: 512 neurones
   - Couche FC2: {len(rewards_eval)} actions (sortie)

3. RÉSULTATS D'ENTRAÎNEMENT
   - Récompense initiale (10 premiers épisodes): {np.mean(rewards_train[:10]):.2f}
   - Récompense finale (10 derniers épisodes): {np.mean(rewards_train[-10:]):.2f}
   - Récompense maximale atteinte: {np.max(rewards_train):.2f}
   - Amélioration: {np.mean(rewards_train[-10:]) - np.mean(rewards_train[:10]):.2f}

4. RÉSULTATS D'ÉVALUATION
   - Nombre d'épisodes d'évaluation: {len(rewards_eval)}
   - Récompense moyenne: {np.mean(rewards_eval):.2f} ± {np.std(rewards_eval):.2f}
   - Récompense minimale: {np.min(rewards_eval):.2f}
   - Récompense maximale: {np.max(rewards_eval):.2f}

5. FICHIERS GÉNÉRÉS
   - Modèle entraîné: {config.MODEL_PATH}
   - Graphiques d'entraînement: training_results.png
   - Vidéo de gameplay: breakout_gameplay.mp4
   - Ce rapport: performance_report.txt

{'='*70}
Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
    
    report_path = os.path.join(config.SAVE_DIR, 'performance_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"Rapport sauvegardé: {report_path}")


def main():
    """
    Fonction principale pour entraîner et évaluer l'agent
    """
    config = Config()
    
    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    print("Début de l'entraînement...")
    agent, rewards_train = train_dqn(config)
    
    print("\nÉvaluation de l'agent entraîné...")
    rewards_eval = evaluate_agent(config.MODEL_PATH, num_episodes=5, save_video=True)
    
    print("\nGénération du rapport de performance...")
    create_performance_report(rewards_train, rewards_eval, config)
    
    print("\n✓ Processus terminé avec succès!")
    print(f"✓ Tous les fichiers sont disponibles dans: {config.SAVE_DIR}")


if __name__ == "__main__":
    main()
