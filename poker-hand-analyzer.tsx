import React, { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";

const PokerHandAnalyzer = () => {
  const [hand, setHand] = useState([
    { suit: '1', rank: '1' },
    { suit: '1', rank: '1' },
    { suit: '1', rank: '1' },
    { suit: '1', rank: '1' },
    { suit: '1', rank: '1' }
  ]);
  const [result, setResult] = useState(null);

  const suits = [
    { value: '1', label: '♥️ Hearts' },
    { value: '2', label: '♠️ Spades' },
    { value: '3', label: '♦️ Diamonds' },
    { value: '4', label: '♣️ Clubs' }
  ];

  const ranks = [
    { value: '1', label: 'Ace' },
    { value: '2', label: '2' },
    { value: '3', label: '3' },
    { value: '4', label: '4' },
    { value: '5', label: '5' },
    { value: '6', label: '6' },
    { value: '7', label: '7' },
    { value: '8', label: '8' },
    { value: '9', label: '9' },
    { value: '10', label: '10' },
    { value: '11', label: 'Jack' },
    { value: '12', label: 'Queen' },
    { value: '13', label: 'King' }
  ];

  const handDescriptions = {
    0: "Nothing in hand",
    1: "One pair - one pair of equal ranks within five cards",
    2: "Two pairs - two pairs of equal ranks within five cards",
    3: "Three of a kind - three equal ranks within five cards",
    4: "Straight - five cards sequentially ranked with no gaps",
    5: "Flush - five cards with the same suit",
    6: "Full house - pair + different rank three of a kind",
    7: "Four of a kind - four equal ranks within five cards",
    8: "Straight flush - straight + flush",
    9: "Royal flush - {Ace, King, Queen, Jack, Ten} + flush"
  };

  const updateHand = (index, field, value) => {
    const newHand = [...hand];
    newHand[index] = { ...newHand[index], [field]: value };
    setHand(newHand);
  };

  const analyzeHand = () => {
    // Convert hand to features similar to the training data
    const features = createFeatures(hand);
    const handType = predictHand(features);
    setResult(handType);
  };

  const createFeatures = (hand) => {
    // Count cards of same rank
    const rankCounts = Array(13).fill(0);
    hand.forEach(card => rankCounts[parseInt(card.rank) - 1]++);
    
    // Count cards of same suit
    const suitCounts = Array(4).fill(0);
    hand.forEach(card => suitCounts[parseInt(card.suit) - 1]++);
    
    // Check if sequential
    const sortedRanks = hand.map(card => parseInt(card.rank)).sort((a, b) => a - b);
    let isSequential = true;
    for (let i = 0; i < sortedRanks.length - 1; i++) {
      if (sortedRanks[i + 1] - sortedRanks[i] !== 1) {
        isSequential = false;
        break;
      }
    }
    // Special case for Ace-low straight
    if (!isSequential && JSON.stringify(sortedRanks) === JSON.stringify([1, 10, 11, 12, 13])) {
      isSequential = true;
    }
    
    // Check flush
    const isFlush = suitCounts.some(count => count === 5);
    
    // Get highest and second highest card counts
    const sortedCounts = [...rankCounts].sort((a, b) => b - a);
    const maxCardCount = sortedCounts[0];
    const secondMaxCardCount = sortedCounts[1];
    
    return {
      rankCounts,
      suitCounts,
      isSequential,
      isFlush,
      maxCardCount,
      secondMaxCardCount
    };
  };

  const predictHand = (features) => {
    const { rankCounts, isSequential, isFlush, maxCardCount, secondMaxCardCount } = features;
    
    // Check for Royal Flush
    const hasRoyalCards = rankCounts[0] === 1 && rankCounts[9] === 1 && 
                         rankCounts[10] === 1 && rankCounts[11] === 1 && rankCounts[12] === 1;
    if (isFlush && hasRoyalCards) return 9;
    
    // Check for Straight Flush
    if (isFlush && isSequential) return 8;
    
    // Check for Four of a Kind
    if (maxCardCount === 4) return 7;
    
    // Check for Full House
    if (maxCardCount === 3 && secondMaxCardCount === 2) return 6;
    
    // Check for Flush
    if (isFlush) return 5;
    
    // Check for Straight
    if (isSequential) return 4;
    
    // Check for Three of a Kind
    if (maxCardCount === 3) return 3;
    
    // Check for Two Pairs
    if (maxCardCount === 2 && secondMaxCardCount === 2) return 2;
    
    // Check for One Pair
    if (maxCardCount === 2) return 1;
    
    // Nothing
    return 0;
  };

  return (
    <Card className="w-full max-w-2xl">
      <CardHeader>
        <CardTitle>Poker Hand Analyzer</CardTitle>
        <CardDescription>Input your five cards to analyze the poker hand</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {hand.map((card, index) => (
            <div key={index} className="flex gap-4 items-center">
              <span className="w-20">Card {index + 1}:</span>
              <Select 
                value={card.suit} 
                onValueChange={(value) => updateHand(index, 'suit', value)}
              >
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {suits.map(suit => (
                    <SelectItem key={suit.value} value={suit.value}>
                      {suit.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              
              <Select 
                value={card.rank} 
                onValueChange={(value) => updateHand(index, 'rank', value)}
              >
                <SelectTrigger className="w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ranks.map(rank => (
                    <SelectItem key={rank.value} value={rank.value}>
                      {rank.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          ))}
          
          <Button onClick={analyzeHand} className="w-full mt-4">
            Analyze Hand
          </Button>
          
          {result !== null && (
            <div className="mt-4 p-4 bg-slate-100 rounded-lg">
              <h3 className="font-bold text-lg mb-2">
                Result: {handDescriptions[result].split(' - ')[0]}
              </h3>
              <p>{handDescriptions[result].split(' - ')[1]}</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
};

export default PokerHandAnalyzer;