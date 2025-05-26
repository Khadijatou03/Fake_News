import pandas as pd
import os

def analyze_dataset(filename, name, sep=','):
    """
    Analyse un dataset de fake news et retourne des statistiques
    
    Args:
        filename (str): Nom du fichier à analyser
        name (str): Nom descriptif du dataset
        sep (str): Séparateur utilisé dans le fichier CSV
        
    Returns:
        DataFrame: Le dataframe chargé ou None en cas d'erreur
    """
    try:
        # Vérifier si le fichier existe localement
        if os.path.exists(filename):
            df = pd.read_csv(filename, sep=sep)
        else:
            # Essayer de charger depuis GitHub
            base_url = "https://raw.githubusercontent.com/Rimka33/detection-de-fake-news/main/"
            url = base_url + filename
            df = pd.read_csv(url, sep=sep)
        
        # Afficher les informations
        print(f"\n=== Dataset: {name} ===")
        print(f"Nombre d'articles: {len(df)}")
        
        # Vérifier si le dataset contient une colonne indiquant si c'est fake ou non
        if 'label' in df.columns:
            fake_count = df[df['label'] == 'FAKE'].shape[0]
            true_count = df[df['label'] == 'REAL'].shape[0]
            print(f"Articles fake: {fake_count}")
            print(f"Articles vrais: {true_count}")
        
        return df
    except Exception as e:
        print(f"Erreur lors de l'analyse du dataset {name}: {str(e)}")
        return None

def get_dataset_stats():
    """
    Récupère les statistiques sur les datasets de fake news
    
    Returns:
        dict: Dictionnaire contenant les statistiques
    """
    stats = {
        'total_articles': 0,
        'total_fake': 0,
        'total_true': 0,
        'datasets': []
    }
    
    # Analyser les datasets anglais
    try:
        fake_df = analyze_dataset('Fake.csv', 'Fake (Anglais)')
        if fake_df is not None:
            stats['total_articles'] += len(fake_df)
            stats['total_fake'] += len(fake_df)
            stats['datasets'].append({
                'name': 'Fake (Anglais)',
                'articles': len(fake_df),
                'fake': len(fake_df),
                'true': 0
            })
        
        true_df = analyze_dataset('True.csv', 'True (Anglais)')
        if true_df is not None:
            stats['total_articles'] += len(true_df)
            stats['total_true'] += len(true_df)
            stats['datasets'].append({
                'name': 'True (Anglais)',
                'articles': len(true_df),
                'fake': 0,
                'true': len(true_df)
            })
    except Exception as e:
        print(f"Erreur lors de l'analyse des datasets anglais: {str(e)}")
    
    # Analyser les datasets français
    try:
        fr_train_df = analyze_dataset('french dataset/train.csv', 'Train (Français)', sep=';')
        if fr_train_df is not None and 'label' in fr_train_df.columns:
            fake_count = fr_train_df[fr_train_df['label'] == 'FAKE'].shape[0]
            true_count = fr_train_df[fr_train_df['label'] == 'REAL'].shape[0]
            
            stats['total_articles'] += len(fr_train_df)
            stats['total_fake'] += fake_count
            stats['total_true'] += true_count
            
            stats['datasets'].append({
                'name': 'Train (Français)',
                'articles': len(fr_train_df),
                'fake': fake_count,
                'true': true_count
            })
        
        fr_test_df = analyze_dataset('french dataset/test.csv', 'Test (Français)', sep=';')
        if fr_test_df is not None and 'label' in fr_test_df.columns:
            fake_count = fr_test_df[fr_test_df['label'] == 'FAKE'].shape[0]
            true_count = fr_test_df[fr_test_df['label'] == 'REAL'].shape[0]
            
            stats['total_articles'] += len(fr_test_df)
            stats['total_fake'] += fake_count
            stats['total_true'] += true_count
            
            stats['datasets'].append({
                'name': 'Test (Français)',
                'articles': len(fr_test_df),
                'fake': fake_count,
                'true': true_count
            })
    except Exception as e:
        print(f"Erreur lors de l'analyse des datasets français: {str(e)}")
    
    return stats

def main():
    print("=== Statistiques sur les datasets de détection de fake news ===\n")
    
    stats = get_dataset_stats()
    
    # Résumé global
    print("\n=== Résumé global ===")
    print(f"Total d'articles: {stats['total_articles']}")
    print(f"Articles fake: {stats['total_fake']}")
    print(f"Articles vrais: {stats['total_true']}")
    
    if stats['total_articles'] > 0:
        fake_percent = (stats['total_fake'] / stats['total_articles']) * 100
        true_percent = (stats['total_true'] / stats['total_articles']) * 100
        print(f"Pourcentage fake: {fake_percent:.1f}%")
        print(f"Pourcentage vrais: {true_percent:.1f}%")

if __name__ == "__main__":
    main()
