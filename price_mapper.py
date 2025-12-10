"""Price Mapper for Mobile Phone Price Prediction
Converts categorical price ranges to actual monetary values based on features
"""

import pandas as pd
import numpy as np

class MobilePriceMapper:
    def __init__(self):
        # Base price ranges for different categories in INR
        self.base_prices = {
            0: (8000, 15000, 'Low Cost 8,000 - 15,000'),
            1: (15000, 30000, 'Medium Cost 15,000 - 30,000'),
            2: (30000, 60000, 'High Cost 30,000 - 60,000'),
            3: (60000, 150000, 'Very High Cost 60,000 - 1,50,000')
        }
        
        # Feature multipliers for price adjustment
        self.feature_weights = {
            'battery_power': 0.15,
            'ram': 0.25,
            'int_memory': 0.20,
            'pc': 0.18,
            'fc': 0.08,
            'px_height': 0.05,
            'px_width': 0.05,
            'clock_speed': 0.04
        }
    
    def calculate_price_range(self, features, price_category):
        """Calculate actual price range based on features and price category"""
        base_min, base_max, label = self.base_prices[price_category]
        
        # Calculate feature-based adjustment
        adjustment_factor = 0
        for feature, weight in self.feature_weights.items():
            if feature in features:
                value = features[feature]
                
                # Normalize feature values for adjustment
                if feature == 'battery_power':
                    normalized = (value - 3000) / 3000  # Battery 3000-6000 mAh - 0-1 scale
                elif feature == 'ram':
                    normalized = (value - 4096) / 12288  # RAM 4096-16384 MB - 0-1 scale
                elif feature == 'int_memory':
                    normalized = (value - 64) / 960  # Storage 64-1024 GB - 0-1 scale
                elif feature == 'pc':
                    normalized = (value - 12) / 96  # Primary camera 12-108 MP - 0-1 scale
                elif feature == 'fc':
                    normalized = (value - 8) / 24  # Front camera 8-32 MP - 0-1 scale
                elif feature == 'px_height':
                    normalized = (value - 720) / 1440  # Screen height 720-2160 - 0-1 scale
                elif feature == 'px_width':
                    normalized = (value - 1280) / 2560  # Screen width 1280-3840 - 0-1 scale
                elif feature == 'clock_speed':
                    normalized = (value - 1.5) / 2.0  # Clock speed 1.5-3.5 GHz - 0-1 scale
                else:
                    normalized = 0
                
                # Apply weight and add to adjustment
                adjustment_factor += normalized * weight
        
        # Calculate price adjustment 0-30% of base price
        price_adjustment = adjustment_factor * 0.3
        
        # Calculate final prices
        adjusted_min = base_min * (1 + price_adjustment)
        adjusted_max = base_max * (1 + price_adjustment)
        
        # Ensure minimum price doesn't go below base minimum
        final_min = max(adjusted_min, base_min)
        final_max = max(adjusted_max, base_max)
        
        return int(final_min), int(final_max)
    
    def get_price_estimate(self, features, price_category):
        """Get price estimate with detailed breakdown"""
        min_price, max_price = self.calculate_price_range(features, price_category)
        avg_price = (min_price + max_price) // 2
        
        price_labels = {
            0: 'Budget',
            1: 'Mid-Range',
            2: 'Premium',
            3: 'Flagship'
        }
        
        return {
            'category': price_labels[price_category],
            'min_price': min_price,
            'max_price': max_price,
            'avg_price': avg_price,
            'price_range': f'Rs.{min_price:,} - Rs.{max_price:,}',
            'estimated_price': f'Rs.{avg_price:,}'
        }
    
    def get_feature_impact(self, features):
        """Get impact of each feature on price"""
        impacts = {}
        for feature, weight in self.feature_weights.items():
            if feature in features:
                value = features[feature]
                
                # Calculate normalized impact
                if feature == 'battery_power':
                    normalized = (value - 3000) / 3000
                elif feature == 'ram':
                    normalized = (value - 4096) / 12288
                elif feature == 'int_memory':
                    normalized = (value - 64) / 960
                elif feature == 'pc':
                    normalized = (value - 12) / 96
                elif feature == 'fc':
                    normalized = (value - 8) / 24
                elif feature == 'px_height':
                    normalized = (value - 720) / 1440
                elif feature == 'px_width':
                    normalized = (value - 1280) / 2560
                elif feature == 'clock_speed':
                    normalized = (value - 1.5) / 2.0
                else:
                    normalized = 0
                
                # Calculate price impact in INR
                price_impact = normalized * weight * 10000  # Scale to reasonable price impact
                impacts[feature] = {
                    'value': value,
                    'impact': int(price_impact),
                    'weight': weight
                }
        
        return impacts
