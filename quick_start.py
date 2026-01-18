#!/usr/bin/env python3
"""
Script de démarrage rapide pour DQN Breakout
Usage: python quick_start.py [--train|--eval|--demo]
"""

import argparse
import os
import sys

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    required = ['gymnasium', 'torch', 'numpy', 'matplotlib', 'cv2', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Dépendances manquantes:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Installation:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ Toutes les dépendances sont installées!")
    return True


def train_mode():
    """Mode entraînement complet"""
    print("\n" + "="*60)
    print("🎮 MODE ENTRAÎNEMENT")
    print("="*60)
    print("\nCe processus va:")
    print("  1. Entraîner l'agent pendant 500 épisodes (~2-3h sur CPU)")
    print("  2. Sauvegarder le modèle automatiquement")
    print("  3. Générer les graphiques de performance")
    print("  4. Créer une vidéo de démonstration")
    print("  5. Produire un rapport détaillé")
    print("\n⚠️  L'entraînement peut prendre du temps. Soyez patient!")
    
    response = input("\nContinuer? (o/n): ")
    if response.lower() != 'o':
        print("Annulé.")
        return
    
    from atari_dqn_breakout import main
    print("\n🚀 Démarrage de l'entraînement...\n")
    main()


def eval_mode():
    """Mode évaluation d'un modèle existant"""
    print("\n" + "="*60)
    print("📊 MODE ÉVALUATION")
    print("="*60)
    
    from atari_dqn_breakout import Config, evaluate_agent
    config = Config()
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ Aucun modèle trouvé à: {config.MODEL_PATH}")
        print("💡 Vous devez d'abord entraîner un modèle avec: python quick_start.py --train")
        return
    
    print(f"\n✅ Modèle trouvé: {config.MODEL_PATH}")
    
    # Choix du mode d'évaluation
    print("\nChoisissez le mode:")
    print("  1. Affichage en temps réel (fenêtre interactive)")
    print("  2. Créer une vidéo MP4")
    print("  3. Les deux")
    
    choice = input("\nVotre choix (1-3): ")
    
    if choice == "1":
        # Affichage en direct
        print("\n🎮 Lancement de l'affichage en temps réel...")
        print("Une fenêtre va s'ouvrir pour montrer le jeu\n")
        rewards = evaluate_agent(config.MODEL_PATH, num_episodes=5, 
                                save_video=False, display_live=True)
    elif choice == "2":
        # Vidéo seulement
        print("\n📹 Création de la vidéo...")
        rewards = evaluate_agent(config.MODEL_PATH, num_episodes=5, 
                                save_video=True, display_live=False)
    elif choice == "3":
        # Les deux
        print("\n🎮 Affichage en direct d'abord...")
        rewards = evaluate_agent(config.MODEL_PATH, num_episodes=3, 
                                save_video=False, display_live=True)
        print("\n📹 Création de la vidéo...")
        evaluate_agent(config.MODEL_PATH, num_episodes=1, 
                      save_video=True, display_live=False)
    else:
        print("Choix invalide, affichage en direct par défaut")
        rewards = evaluate_agent(config.MODEL_PATH, num_episodes=5, 
                                save_video=False, display_live=True)
    
    print("\n" + "="*60)
    print("📈 RÉSULTATS")
    print("="*60)
    print(f"Récompense moyenne: {sum(rewards)/len(rewards):.2f}")
    print(f"Meilleure performance: {max(rewards):.2f}")
    print(f"Pire performance: {min(rewards):.2f}")


def play_mode():
    """Mode jeu en direct simplifié"""
    print("\n" + "="*60)
    print("🎮 MODE JEU EN DIRECT")
    print("="*60)
    
    from atari_dqn_breakout import Config, play_live
    config = Config()
    
    if not os.path.exists(config.MODEL_PATH):
        print(f"\n❌ Aucun modèle trouvé à: {config.MODEL_PATH}")
        print("💡 Vous devez d'abord entraîner un modèle")
        return
    
    print(f"\n✅ Modèle trouvé: {config.MODEL_PATH}")
    print("\nCombien d'épisodes voulez-vous voir ?")
    
    try:
        num_ep = int(input("Nombre d'épisodes (1-10, défaut=3): ") or "3")
        num_ep = max(1, min(10, num_ep))
    except:
        num_ep = 3
    
    print("\nVitesse d'affichage:")
    print("  1. Normale (30 FPS)")
    print("  2. Rapide (60 FPS)")
    print("  3. Très rapide (pas de limite)")
    
    speed = input("\nChoix (1-3, défaut=1): ") or "1"
    
    fps_map = {"1": 30, "2": 60, "3": None}
    fps = fps_map.get(speed, 30)
    
    play_live(config.MODEL_PATH, num_episodes=num_ep, fps_limit=fps)


def demo_mode():
    """Mode démonstration rapide (entraînement court)"""
    print("\n" + "="*60)
    print("🎯 MODE DÉMONSTRATION RAPIDE")
    print("="*60)
    print("\nEntraînement rapide de 50 épisodes (~15-20 min)")
    print("⚠️  Les performances seront limitées mais vous verrez le processus\n")
    
    response = input("Continuer? (o/n): ")
    if response.lower() != 'o':
        print("Annulé.")
        return
    
    from atari_dqn_breakout import train_dqn, evaluate_agent, Config, create_performance_report
    import os
    
    # Configuration réduite pour démonstration
    config = Config()
    config.NUM_EPISODES = 50  # Réduit pour démonstration
    config.LEARNING_START = 1000  # Commence l'apprentissage plus tôt
    
    print("\n🚀 Démarrage de la démonstration...\n")
    
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    agent, rewards_train = train_dqn(config)
    
    print("\nÉvaluation du modèle...")
    rewards_eval = evaluate_agent(config.MODEL_PATH, num_episodes=3, save_video=True)
    
    create_performance_report(rewards_train, rewards_eval, config)
    
    print("\n✅ Démonstration terminée!")


def main():
    parser = argparse.ArgumentParser(
        description="DQN Atari Breakout - Script de démarrage rapide",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python quick_start.py --train     # Entraînement complet (500 épisodes)
  python quick_start.py --eval      # Évaluation d'un modèle existant
  python quick_start.py --demo      # Démonstration rapide (50 épisodes)
  
Sans argument, mode interactif.
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='Entraînement complet')
    parser.add_argument('--eval', action='store_true',
                       help='Évaluation du modèle')
    parser.add_argument('--play', action='store_true',
                       help='Jeu en direct (affichage temps réel)')
    parser.add_argument('--demo', action='store_true',
                       help='Démonstration rapide')
    
    args = parser.parse_args()
    
    # Vérifier les dépendances
    if not check_dependencies():
        sys.exit(1)
    
    # Mode ligne de commande
    if args.train:
        train_mode()
    elif args.eval:
        eval_mode()
    elif args.play:
        play_mode()
    elif args.demo:
        demo_mode()
    else:
        # Mode interactif
        print("\n" + "="*60)
        print("🎮 DQN ATARI BREAKOUT - Menu Principal")
        print("="*60)
        print("\nChoisissez un mode:")
        print("  1. Entraînement complet (500 épisodes, ~2-3h)")
        print("  2. Évaluation d'un modèle existant")
        print("  3. Jeu en direct (affichage temps réel)")
        print("  4. Démonstration rapide (50 épisodes, ~20min)")
        print("  5. Quitter")
        
        choice = input("\nVotre choix (1-5): ")
        
        if choice == '1':
            train_mode()
        elif choice == '2':
            eval_mode()
        elif choice == '3':
            play_mode()
        elif choice == '4':
            demo_mode()
        elif choice == '5':
            print("Au revoir!")
        else:
            print("Choix invalide.")


if __name__ == "__main__":
    main()
