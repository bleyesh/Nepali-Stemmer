import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class NepaliStemmer:
    """    
    The algorithm works in multiple stages:
    1. Normalization and preprocessing
    2. Statistical suffix frequency checking
    3. Rule-based suffix stripping (with priority ordering)
    4. Validation using minimum stem length and edit distance
    """
    
    def __init__(self):
        # Minimum stem length (in characters)
        self.min_stem_length = 2
        
        # Initialize suffix categories with frequency weights
        # Higher weight = higher priority for removal
        self.initialize_suffix_rules()
        
        # Common Nepali words that should not be stemmed
        self.stopwords = self.load_stopwords()
        
        # Edit distance threshold for validation
        self.edit_distance_threshold = 0.4  # 40% of word length
        
    def initialize_suffix_rules(self):
        """
        Initialize Nepali suffix rules organized by morphological categories.
        Suffixes are ordered by priority (longer, more specific first).
        """
        
        self.case_markers = {
            'हरूलाई': 5.0,
            'हरूको': 5.0,
            'हरूमा': 5.0,
            'हरूबाट': 5.0,
            'हरूसँग': 5.0,
            'लाई': 4.0,
            'को': 4.0,
            'मा': 4.0,
            'बाट': 4.0,
            'सँग': 4.0,
            'द्वारा': 4.0,
            'का': 3.5,
            'की': 3.5,
            'ले': 3.5,
        }
        
        self.plural_markers = {
            'हरू': 4.5,
            'हरु': 4.5,
        }
        
        self.verbal_suffixes = {
            'एका': 4.0,
            'एको': 4.0,
            'एकी': 4.0,
            'इएको': 3.8,
            'दैछ': 3.5,
            'दैछन्': 3.5,
            'दछ': 3.5,
            'दछन्': 3.5,
            'यो': 3.5,
            'थियो': 3.5,
            'थिए': 3.5,
            'न्छ': 3.0,
            'न्छन्': 3.0,
            'ने': 3.0,
            'नु': 3.0,
            'ला': 3.0,
            'उ': 2.5,
            'छ': 2.5,
            'छन्': 2.5,
            'छु': 2.5,
            'औं': 2.5,
            'ओ': 2.5,
        }
        
        self.adjectival_suffixes = {
            'इलो': 3.0,
            'इली': 3.0,
            'इला': 3.0,
            'पूर्ण': 2.5,
            'हीन': 2.5,
        }
        
        self.nominal_suffixes = {
            'पन': 3.0,
            'पना': 3.0,
            'ता': 2.5,
            'त्व': 2.5,
            'इक': 2.5,
            'वाला': 2.5,
            'दार': 2.5,
        }
    
    def load_stopwords(self) -> Set[str]:
        """Load common Nepali stopwords that should not be stemmed."""
        return {
            'छ', 'छन्', 'हो', 'होइन', 'हुन्', 'थियो', 'थिए',
            'र', 'वा', 'तर', 'पनि', 'नि', 'त', 'भने',
            'को', 'का', 'की', 'ले', 'लाई', 'मा', 'बाट',
            'यो', 'यी', 'त्यो', 'ती', 'उ', 'उनी',
            'म', 'हामी', 'तिमी', 'तपाईं', 'उनीहरू',
        }
    
    def normalize(self, word: str) -> str:
        """
        Normalize Nepali text.
        - Remove extra whitespace
        - Normalize unicode variations
        """
        word = word.strip()
        # Normalize common unicode variations
        word = word.replace('़', '')  # Remove nukta if needed for stemming
        return word
    
    def calculate_edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate minimum edit distance (Levenshtein distance) between two strings.
        Used for validation of stemming operations.
        """
        m, n = len(s1), len(s2)
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],
                        dp[i][j-1],
                        dp[i-1][j-1]
                    )
        
        return dp[m][n]
    
    def validate_stem(self, original: str, stem: str) -> bool:
        """
        Validate that the stem is reasonable using multiple criteria:
        1. Minimum length check
        2. Edit distance check
        3. Not over-stemmed
        """
        # Check minimum length
        if len(stem) < self.min_stem_length:
            return False
        
        # Check edit distance (stem shouldn't be too different)
        max_distance = int(len(original) * self.edit_distance_threshold)
        if self.calculate_edit_distance(original, stem) > max_distance:
            return False
        
        # Stem should not be too short relative to original
        if len(stem) < len(original) * 0.3:  # At least 30% of original
            return False
        
        return True
    
    def get_all_suffixes_sorted(self) -> List[Tuple[str, float]]:
        """
        Get all suffixes sorted by priority (frequency * length).
        Longer, more frequent suffixes are tried first.
        """
        all_suffixes = {}
        
        # Collect all suffixes with their weights
        for suffix, weight in self.case_markers.items():
            all_suffixes[suffix] = weight
        for suffix, weight in self.plural_markers.items():
            all_suffixes[suffix] = weight
        for suffix, weight in self.verbal_suffixes.items():
            all_suffixes[suffix] = weight
        for suffix, weight in self.adjectival_suffixes.items():
            all_suffixes[suffix] = weight
        for suffix, weight in self.nominal_suffixes.items():
            all_suffixes[suffix] = weight
        
        # Sort by priority: weight * length (prefer longer, weighted suffixes)
        sorted_suffixes = sorted(
            all_suffixes.items(),
            key=lambda x: x[1] * len(x[0]),
            reverse=True
        )
        
        return sorted_suffixes
    
    def strip_suffix(self, word: str, suffix: str) -> str:
        """Strip a suffix from the word if it ends with it."""
        if word.endswith(suffix):
            return word[:-len(suffix)]
        return word
    
    def apply_sandhi_rules(self, stem: str) -> str:
        """
        Apply reverse sandhi rules to restore proper stem form.
        Sandhi is the phonological modification at morpheme boundaries.
        """
        # Common stem corrections after suffix removal
        
        # If stem ends with doubled consonant, reduce it
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            # But keep doubled consonants in common patterns
            if stem[-1] not in ['त्त', 'द्द', 'न्न']:
                stem = stem[:-1]
        
        return stem
    
    def stem(self, word: str, apply_validation: bool = True) -> str:
        """
        Main stemming function - returns the stem of a Nepali word.
        
        Args:
            word: Input Nepali word
            apply_validation: Whether to validate stems (default True)
            
        Returns:
            Stemmed word
        """
        # Normalize input
        original_word = word
        word = self.normalize(word)
        
        # Don't stem very short words
        if len(word) <= 2:
            return word
        
        # Don't stem stopwords
        if word in self.stopwords:
            return word
        
        # Get sorted suffixes
        sorted_suffixes = self.get_all_suffixes_sorted()
        
        # Try to remove suffixes (greedy approach - remove longest matching first)
        best_stem = word
        best_score = 0
        
        for suffix, weight in sorted_suffixes:
            if word.endswith(suffix):
                potential_stem = self.strip_suffix(word, suffix)
                
                # Apply sandhi rules
                potential_stem = self.apply_sandhi_rules(potential_stem)
                
                # Validate the stem
                if apply_validation:
                    if self.validate_stem(word, potential_stem):
                        # Calculate score based on suffix weight and length
                        score = weight * len(suffix)
                        if score > best_score:
                            best_stem = potential_stem
                            best_score = score
                else:
                    # Without validation, take first match
                    return potential_stem
        
        # If no valid suffix found, return original
        if best_stem == word:
            return word
        
        return best_stem
    
    def get_suffix_info(self, word: str) -> Dict:
        """
        Get detailed information about suffix removal for analysis.
        
        Args:
            word: Input Nepali word
            
        Returns:
            Dictionary with stemming details
        """
        original_word = word
        word = self.normalize(word)
        stem = self.stem(word)
        
        removed_suffix = ""
        if word != stem:
            removed_suffix = word[len(stem):]
        
        return {
            'original': original_word,
            'normalized': word,
            'stem': stem,
            'suffix_removed': removed_suffix,
            'edit_distance': self.calculate_edit_distance(word, stem),
            'stem_length': len(stem),
            'original_length': len(word)
        }


def main():
    """
    Main function demonstrating the Nepali stemmer usage.
    """
    # Initialize the stemmer
    stemmer = NepaliStemmer()
    
    test_words = [
    'किताबहरू',
    'किताबलाई',
    'किताबको',
    'घरहरू',
    'घरको',
    'लेख्यो',
    'लेख्नेछ',
    'बोल्दै',
    'हिँड्नेछ',
    'सफलता',
    'गरिबी',
    'विद्यार्थीहरू',
    'विद्यार्थीलाई',
    'नेपाललाई',
    'फूलको',
    'कालेले',
    'लेख्छु',
    'लेख्यो',
    'सुन्दरता',
    'शिक्षकको',
    'किसानको',
    'लेखकको',
    'पढिरहेको',
    'बोल्नेछ',
    'खानेछ',
    'मिठास',
    'सेवक',
    'हिँड्नेछ',
    'मान्छेहरू',
    'घरहरू',
    'किताबबाट',
    'राष्ट्रमा',
    'सम्पदाको',
    'तिमी',
    'लिँदै',
    'सम्पदाको',
    'सम्पदाहरू',
    'गारेकी',
    'रोएको',
    'गर्यो',
    'गुलियो',
    'गर्नेछ',
    'मान्छेको',
    'चिप्लनु',
    'गारेकी',
    'सुन्दरता',
    'साहसिकता',
    'बहिनीलाई',
    'केटीकी',
    'सेविकाकी'
    ]
    
    print("Test Cases:")
    print("-" * 70)
    print(f"{'Original Word':<25} {'Stem':<20} {'Suffix Removed':<15}")
    print("-" * 70)
    file=open("output_root.txt", "w", encoding="utf-8")
    file_suffix=open("output_suffix.txt", "w", encoding="utf-8")
    for word in test_words:
        info = stemmer.get_suffix_info(word)
        print(f"{info['original']:<25} {info['stem']:<20} {info['suffix_removed']:<15}")
        file.write(f"{info['stem']}\n")
        file_suffix.write(f"{info['suffix_removed']}\n")
    file.close()
    file_suffix.close()

    print("Interactive Mode:")
    print("Enter Nepali words to stem (or 'quit' to exit)")
    print("-" * 70)
    
    while True:
        try:
            user_input = input("\nEnter word: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not user_input:
                continue
            
            info = stemmer.get_suffix_info(user_input)
            
            print(f"\n  Original:        {info['original']}")
            print(f"  Stem:            {info['stem']}")
            print(f"  Suffix removed:  {info['suffix_removed']}")
            print(f"  Edit distance:   {info['edit_distance']}")
            print(f"  Stem ratio:      {info['stem_length']}/{info['original_length']} " +
                  f"({info['stem_length']/info['original_length']*100:.1f}%)")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
