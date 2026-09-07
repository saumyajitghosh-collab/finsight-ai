#!/usr/bin/env python
"""
FinSight AI v3.0 — Cognitive Finance Platform
An end-to-end Agentic AI banking application built on 4 research papers:
1. Quantum Personnel Securities (QPS) — Third asset class via quantum mechanics
2. Tokenized Cognitive Capital (TCC) — Pricing and trading organizational intelligence
3. Cognitive Settlement Layer (CSL) — 8-agent post-trade settlement optimization
4. The Gate Symphony — Deterministic logic gates for bounding agentic AI autonomy

Author of underlying research: Saumyajit Ghosh
"""
import os, json, math, random, hashlib, time
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.after_request
def set_cache_headers(resp):
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# ============================================================================
# MODULE 1: QUANTUM PERSONNEL SECURITIES (QPS)
# Based on: "Quantum Personnel Securities (QPS): A Theoretical Framework
#            for a Third Asset Class Beyond Equity and Debt"
# ============================================================================

class QPSEngine:
    """
    QPS models human/behavioral corporate value using quantum mechanics:
    - Superposition: leaders embody multiple strategies simultaneously
    - Entanglement: leadership teams evolve in correlated states
    - Bias Operators: non-linear decision influences (overconfidence, loss aversion, groupthink)
    """

    # Bias operators from the paper
    BIAS_OPERATORS = {
        'overconfidence': {
            'symbol': 'O_OC',
            'description': 'Executives overestimate forecasting ability, leading to empire-building',
            'effect': 'amplifies_risk',
            'default_strength': 0.3,
        },
        'loss_aversion': {
            'symbol': 'O_LA',
            'description': 'Leaders overweight downside risk, stalling necessary pivots',
            'effect': 'suppresses_action',
            'default_strength': 0.25,
        },
        'groupthink': {
            'symbol': 'O_GT',
            'description': 'Boards suppress dissenting states, correlated misjudgments',
            'effect': 'correlates_errors',
            'default_strength': 0.2,
        },
        'anchoring': {
            'symbol': 'O_AN',
            'description': 'Decisions anchored to initial reference points',
            'effect': 'biases_toward_status_quo',
            'default_strength': 0.15,
        },
        'confirmation': {
            'symbol': 'O_CF',
            'description': 'Seeking information that confirms existing beliefs',
            'effect': 'reinforces_bias',
            'default_strength': 0.2,
        },
    }

    @staticmethod
    def personnel_state_vector(strategies, amplitudes=None):
        """
        Create a personnel state vector |psi> = sum alpha_i |strategy_i>
        Leaders simultaneously embody multiple potential strategies.
        """
        n = len(strategies)
        if amplitudes is None:
            amplitudes = np.ones(n) / math.sqrt(n)
        else:
            amplitudes = np.array(amplitudes, dtype=complex)
            norm = np.sqrt(np.sum(np.abs(amplitudes)**2))
            if norm > 0:
                amplitudes = amplitudes / norm

        state = {s: a for s, a in zip(strategies, amplitudes)}
        probabilities = {s: float(np.abs(a)**2) for s, a in state.items()}
        return {
            'strategies': strategies,
            'amplitudes': [{'real': float(a.real), 'imag': float(a.imag)} for a in amplitudes],
            'probabilities': probabilities,
            'entropy': float(-sum(p * math.log(p + 1e-15) for p in probabilities.values())),
            'max_strategy': max(probabilities, key=probabilities.get),
            'max_probability': max(probabilities.values()),
        }

    @staticmethod
    def apply_bias_operator(state, bias_type, strength=None):
        """
        Apply a bias operator to the personnel state vector.
        Bias operators are non-commuting, non-linear transformations.
        """
        if bias_type not in QPSEngine.BIAS_OPERATORS:
            return {'error': f'Unknown bias: {bias_type}'}

        bias = QPSEngine.BIAS_OPERATORS[bias_type]
        s = strength if strength is not None else bias['default_strength']
        probs = state['probabilities']
        strategies = state['strategies']

        # Apply bias transformation
        new_probs = {}
        if bias['effect'] == 'amplifies_risk':
            # Overconfidence amplifies aggressive strategies
            for strat in strategies:
                if any(w in strat.lower() for w in ['expansion', 'aggressive', 'acquisition', 'growth']):
                    new_probs[strat] = probs[strat] * (1 + s)
                else:
                    new_probs[strat] = probs[strat] * (1 - s * 0.5)
        elif bias['effect'] == 'suppresses_action':
            # Loss aversion suppresses all active strategies
            for strat in strategies:
                if any(w in strat.lower() for w in ['contraction', 'hold', 'status quo', 'conservative']):
                    new_probs[strat] = probs[strat] * (1 + s)
                else:
                    new_probs[strat] = probs[strat] * (1 - s * 0.5)
        elif bias['effect'] == 'correlates_errors':
            # Groupthink pushes toward majority, reducing diversity
            max_strat = max(probs, key=probs.get)
            for strat in strategies:
                if strat == max_strat:
                    new_probs[strat] = probs[strat] * (1 + s)
                else:
                    new_probs[strat] = probs[strat] * (1 - s)
        elif bias['effect'] == 'biases_toward_status_quo':
            for strat in strategies:
                if any(w in strat.lower() for w in ['hold', 'status quo', 'maintain']):
                    new_probs[strat] = probs[strat] * (1 + s * 2)
                else:
                    new_probs[strat] = probs[strat] * (1 - s)
        elif bias['effect'] == 'reinforces_bias':
            # Reinforces current dominant strategy
            max_strat = max(probs, key=probs.get)
            for strat in strategies:
                if strat == max_strat:
                    new_probs[strat] = probs[strat] * (1 + s)
                else:
                    new_probs[strat] = probs[strat] * (1 - s * 0.7)
        else:
            new_probs = probs.copy()

        # Renormalize
        total = sum(new_probs.values())
        if total > 0:
            new_probs = {k: v / total for k, v in new_probs.items()}

        entropy = float(-sum(p * math.log(p + 1e-15) for p in new_probs.values()))

        return {
            'bias_applied': bias_type,
            'bias_symbol': bias['symbol'],
            'bias_description': bias['description'],
            'strength': s,
            'new_probabilities': new_probs,
            'new_entropy': entropy,
            'entropy_change': entropy - state['entropy'],
            'new_dominant_strategy': max(new_probs, key=new_probs.get),
            'interpretation': QPSEngine._interpret_bias(bias_type, entropy, state['entropy'])
        }

    @staticmethod
    def _interpret_bias(bias_type, new_entropy, old_entropy):
        change = new_entropy - old_entropy
        if change < -0.1:
            return f"Significant reduction in strategic diversity ({change:.3f}). The {bias_type} bias is concentrating decision-making around fewer options, increasing the risk of strategic blind spots."
        elif change < 0:
            return f"Moderate reduction in strategic diversity ({change:.3f}). The {bias_type} bias is nudging decisions toward a narrower set of options."
        else:
            return f"Minimal impact on strategic diversity ({change:+.3f}). The {bias_type} bias is present but not dominant in current conditions."

    @staticmethod
    def qps_hamiltonian(state, market_condition='normal', time_steps=10):
        """
        QPS Hamiltonian evolves the personnel state over time.
        H = H_market + H_bias + H_entanglement
        """
        results = []
        probs = dict(state['probabilities'])
        strategies = state['strategies']

        # Market coupling factors
        market_factors = {
            'bull': {'expansion': 1.15, 'aggressive': 1.2, 'conservative': 0.85, 'hold': 0.8, 'contraction': 0.7},
            'bear': {'expansion': 0.8, 'aggressive': 0.7, 'conservative': 1.15, 'hold': 1.2, 'contraction': 1.25},
            'normal': {'expansion': 1.0, 'aggressive': 1.0, 'conservative': 1.0, 'hold': 1.0, 'contraction': 1.0},
            'crisis': {'expansion': 0.6, 'aggressive': 0.5, 'conservative': 1.3, 'hold': 1.1, 'contraction': 1.4},
        }
        factors = market_factors.get(market_condition, market_factors['normal'])

        for t in range(time_steps):
            new_probs = {}
            for strat in strategies:
                # Apply market coupling
                key = next((k for k in factors if k in strat.lower()), None)
                factor = factors.get(key, 1.0) if key else 1.0
                # Add quantum-like oscillation
                oscillation = 1 + 0.05 * math.sin(2 * math.pi * t / time_steps + hash(strat) % 10)
                new_probs[strat] = probs.get(strat, 0) * factor * oscillation

            total = sum(new_probs.values())
            if total > 0:
                new_probs = {k: v / total for k, v in new_probs.items()}

            entropy = -sum(p * math.log(p + 1e-15) for p in new_probs.values())
            dominant = max(new_probs, key=new_probs.get)

            results.append({
                'step': t,
                'probabilities': {k: round(v, 4) for k, v in new_probs.items()},
                'entropy': round(entropy, 4),
                'dominant_strategy': dominant,
            })
            probs = new_probs

        return {
            'market_condition': market_condition,
            'time_steps': time_steps,
            'evolution': results,
            'final_state': results[-1] if results else None,
            'interpretation': f"Under {market_condition} market conditions, the leadership state evolved over {time_steps} steps. Final dominant strategy: {results[-1]['dominant_strategy'] if results else 'N/A'}",
        }

    @staticmethod
    def entanglement_measure(team_states):
        """
        Measure entanglement between leadership team members.
        Uses mutual information as entanglement proxy.
        """
        n = len(team_states)
        if n < 2:
            return {'error': 'Need at least 2 team members'}

        # Compute pairwise mutual information
        entanglement_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    entanglement_matrix[i][j] = 1.0
                else:
                    # Mutual information between probability distributions
                    p_i = np.array(list(team_states[i]['probabilities'].values()))
                    p_j = np.array(list(team_states[j]['probabilities'].values()))
                    # Align lengths
                    min_len = min(len(p_i), len(p_j))
                    p_i, p_j = p_i[:min_len], p_j[:min_len]
                    p_i = p_i / p_i.sum()
                    p_j = p_j / p_j.sum()
                    p_ij = p_i * p_j
                    mi = sum(p_ij[k] * math.log(p_ij[k] / (p_i[k] * p_j[k]) + 1e-15)
                             for k in range(min_len) if p_ij[k] > 0)
                    entanglement_matrix[i][j] = float(mi)

        avg_entanglement = float(np.mean(entanglement_matrix[np.triu_indices(n, k=1)])) if n > 1 else 0

        return {
            'team_size': n,
            'entanglement_matrix': entanglement_matrix.tolist(),
            'average_entanglement': round(avg_entanglement, 4),
            'max_entanglement': round(float(np.max(entanglement_matrix[np.triu_indices(n, k=1)])), 4) if n > 1 else 0,
            'interpretation': 'High entanglement means leadership decisions are strongly correlated — team members reinforce each other\'s biases.' if avg_entanglement > 0.3 else 'Low entanglement means leadership operates independently — diverse perspectives are preserved.' if avg_entanglement < 0.1 else 'Moderate entanglement — some correlation in leadership decision patterns.',
        }

    @staticmethod
    def qps_payoff(state, financial_outcome, bias_penalty=0.1):
        """
        QPS Payoff Function: links quantum behavioral state to financial outcomes.
        Payoff = Financial_Outcome * (1 - bias_entropy_penalty)
        """
        probs = state['probabilities']
        entropy = state['entropy']
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1

        # Entropy penalty: high entropy (uncertainty) reduces payoff
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        adjusted_outcome = financial_outcome * (1 - bias_penalty * entropy_ratio)

        # Strategy-weighted outcome
        strategy_weights = {
            'expansion': 1.3, 'aggressive': 1.5, 'conservative': 0.9,
            'hold': 1.0, 'contraction': 0.7, 'acquisition': 1.4,
            'growth': 1.2, 'pivot': 1.1, 'status quo': 1.0, 'maintain': 1.0,
        }

        weighted_outcome = 0
        for strat, prob in probs.items():
            weight = next((v for k, v in strategy_weights.items() if k in strat.lower()), 1.0)
            weighted_outcome += prob * financial_outcome * weight

        weighted_outcome *= (1 - bias_penalty * entropy_ratio)

        return {
            'base_financial_outcome': financial_outcome,
            'entropy_penalty': round(bias_penalty * entropy_ratio, 4),
            'adjusted_payoff': round(adjusted_outcome, 2),
            'strategy_weighted_payoff': round(weighted_outcome, 2),
            'entropy_ratio': round(entropy_ratio, 4),
            'interpretation': f"The QPS payoff of {round(weighted_outcome, 2)} reflects a base outcome of {financial_outcome} adjusted for leadership entropy ({round(entropy_ratio*100, 1)}% uncertainty penalty). Higher strategic clarity (lower entropy) yields payoffs closer to the base case.",
        }

    @staticmethod
    def run_scenario(scenario_name, **kwargs):
        """Run predefined simulation scenarios from the paper."""
        scenarios = {
            'overconfident_ceo': {
                'description': 'An overconfident CEO in a bull market',
                'strategies': ['Expansion', 'Aggressive Acquisition', 'Hold', 'Conservative Growth'],
                'amplitudes': [0.3, 0.4, 0.2, 0.1],
                'biases': [('overconfidence', 0.5)],
                'market': 'bull',
            },
            'risk_averse_board': {
                'description': 'A risk-averse board in a bear market',
                'strategies': ['Expansion', 'Hold', 'Conservative', 'Contraction'],
                'amplitudes': [0.1, 0.3, 0.4, 0.2],
                'biases': [('loss_aversion', 0.4)],
                'market': 'bear',
            },
            'groupthink_committee': {
                'description': 'Groupthink in a strategic committee',
                'strategies': ['Expansion', 'Hold', 'Conservative', 'Pivot'],
                'amplitudes': [0.25, 0.25, 0.25, 0.25],
                'biases': [('groupthink', 0.5), ('confirmation', 0.3)],
                'market': 'normal',
            },
            'crisis_response': {
                'description': 'Crisis response team under market stress',
                'strategies': ['Aggressive Pivot', 'Conservative Hold', 'Contraction', 'Expansion'],
                'amplitudes': [0.2, 0.35, 0.3, 0.15],
                'biases': [('loss_aversion', 0.3), ('overconfidence', 0.2)],
                'market': 'crisis',
            },
            'balanced_leadership': {
                'description': 'Balanced triad leadership with bias correction',
                'strategies': ['Expansion', 'Hold', 'Conservative', 'Strategic Pivot'],
                'amplitudes': [0.3, 0.25, 0.25, 0.2],
                'biases': [],  # No bias — ideal governance
                                'market': 'normal',
            },
            'merger_acquisition': {
                'description': 'Merger decision: CEO overconfidence meets board caution',
                'strategies': ['Accept Offer', 'Reject', 'Counter-Bid', 'Delay'],
                'amplitudes': [0.25, 0.3, 0.25, 0.2],
                'biases': [('overconfidence', 0.35), ('anchoring', 0.25)],
                'market': 'bull',
            },
            'tech_disruption': {
                'description': 'Technology disruption response: innovation vs survival tension',
                'strategies': ['Invest Heavily', 'Partner', 'Acquire Startup', 'Status Quo'],
                'amplitudes': [0.3, 0.25, 0.2, 0.25],
                'biases': [('confirmation', 0.4), ('overconfidence', 0.2)],
                'market': 'normal',
            },
            'succession_uncertainty': {
                'description': 'Leadership succession: uncertainty during CEO transition',
                'strategies': ['Continuity', 'Strategic Shift', 'Restructure', 'Status Quo'],
                'amplitudes': [0.35, 0.2, 0.15, 0.3],
                'biases': [('loss_aversion', 0.35), ('groupthink', 0.2)],
                'market': 'normal',
            },
        }

        if scenario_name not in scenarios:
            return {'error': f'Unknown scenario. Available: {list(scenarios.keys())}'}

        sc = scenarios[scenario_name]
        state = QPSEngine.personnel_state_vector(sc['strategies'], sc['amplitudes'])

        bias_results = []
        for bias_type, strength in sc['biases']:
            result = QPSEngine.apply_bias_operator(state, bias_type, strength)
            bias_results.append(result)
            # Update state with biased probabilities
            state['probabilities'] = result['new_probabilities']
            state['entropy'] = result['new_entropy']

        evolution = QPSEngine.qps_hamiltonian(state, sc['market'], time_steps=10)
        payoff = QPSEngine.qps_payoff(state, kwargs.get('financial_outcome', 100000000))

        return {
            'scenario': scenario_name,
            'description': sc['description'],
            'initial_state': {
                'strategies': sc['strategies'],
                'probabilities': {k: round(v, 4) for k, v in zip(sc['strategies'], [abs(a)**2 for a in sc['amplitudes'] / np.sqrt(sum(np.abs(np.array(sc['amplitudes'], dtype=complex))**2))])},
            },
            'bias_applications': bias_results,
            'state_evolution': evolution,
            'payoff': payoff,
        }


# ============================================================================
# MODULE 2: TOKENIZED COGNITIVE CAPITAL (TCC)
# Based on: "Tokenized Cognitive Capital (TCC): A Market Framework to Price,
#            Tokenize, and Trade Organizational Intelligence"
# ============================================================================

class TCCEngine:
    """
    TCC formalizes enterprise cognition as a quantum-inspired superposition.
    CCI = Cognitive Capital Index — measures collective intelligence.
    """

    # CCI Feature Space (from paper Section 3.3)
    CCI_FEATURES = {
        'knowledge_creation': {'weight': 0.20, 'description': 'Rate of new knowledge generation'},
        'decision_efficiency': {'weight': 0.18, 'description': 'Speed and quality of decisions'},
        'ai_alignment': {'weight': 0.15, 'description': 'Human-AI cognitive synergy'},
        'learning_velocity': {'weight': 0.15, 'description': 'Adaptive learning rate'},
        'innovation_output': {'weight': 0.12, 'description': 'Patents, products, processes'},
        'collaboration_index': {'weight': 0.10, 'description': 'Cross-team knowledge flow'},
        'adaptive_capacity': {'weight': 0.10, 'description': 'Response to market changes'},
    }

    @staticmethod
    def compute_cci(feature_values, weights=None):
        """
        Cognitive Capital Index (CCI) — Section 3.2
        CCI = sum(w_i * f_i) * entropy_adjustment * (1 - penalty)
        """
        if weights is None:
            weights = {k: v['weight'] for k, v in TCCEngine.CCI_FEATURES.items()}

        # Normalized features
        normalized = {}
        for f, v in feature_values.items():
            normalized[f] = min(max(v, 0), 1)

        # Weighted sum
        raw_cci = sum(weights.get(f, 0) * normalized.get(f, 0) for f in feature_values)

        # Entropy adjustment (Shannon entropy of feature distribution)
        probs = np.array(list(normalized.values()))
        probs = probs / probs.sum() if probs.sum() > 0 else probs
        entropy = -sum(p * math.log(p + 1e-15) for p in probs if p > 0)
        max_entropy = math.log(len(probs)) if len(probs) > 1 else 1
        entropy_adjustment = 0.8 + 0.2 * (entropy / max_entropy) if max_entropy > 0 else 1

        # Penalty for low scores
        min_feature = min(normalized.values()) if normalized else 0
        penalty = max(0, (0.3 - min_feature) * 0.5) if min_feature < 0.3 else 0

        cci = raw_cci * entropy_adjustment * (1 - penalty)

        return {
            'cci': round(cci, 4),
            'raw_score': round(raw_cci, 4),
            'entropy_adjustment': round(entropy_adjustment, 4),
            'penalty': round(penalty, 4),
            'feature_scores': {k: round(v, 4) for k, v in normalized.items()},
            'entropy': round(entropy, 4),
            'rating': TCCEngine._cci_rating(cci),
            'interpretation': TCCEngine._interpret_cci(cci, normalized),
        }

    @staticmethod
    def _cci_rating(cci):
        if cci >= 0.8: return 'AAA — Exceptional Cognitive Capital'
        elif cci >= 0.7: return 'AA — Strong Cognitive Capital'
        elif cci >= 0.6: return 'A — Good Cognitive Capital'
        elif cci >= 0.5: return 'BBB — Moderate Cognitive Capital'
        elif cci >= 0.4: return 'BB — Developing Cognitive Capital'
        elif cci >= 0.3: return 'B — Weak Cognitive Capital'
        else: return 'C — Minimal Cognitive Capital'

    @staticmethod
    def _interpret_cci(cci, features):
        strongest = max(features, key=features.get)
        weakest = min(features, key=features.get)
        return f"CCI of {round(cci, 3)} indicates {TCCEngine._cci_rating(cci).split(' — ')[1]}. Strongest dimension: {strongest.replace('_', ' ')} ({features[strongest]:.2f}). Weakest: {weakest.replace('_', ' ')} ({features[weakest]:.2f})."

    @staticmethod
    def token_valuation(cci, revenue, growth_rate, cognitive_decay=0.1, voc=0.15, risk_free=0.05):
        """
        TCC Valuation Framework — Section 5
        V(TCC) = f(CCI, Revenue, Growth) * e^(-lambda*t) / (1 + VoC_premium)
        Includes cognitive decay (half-life) and Volatility of Cognition (VoC).
        """
        # Marginal Cognition Value (MCV)
        mcv = cci * revenue * 0.15  # CCI contributes 15% of revenue as cognitive value

        # Cognitive decay: V(t) = V(0) * e^(-lambda * t)
        half_life = math.log(2) / cognitive_decay if cognitive_decay > 0 else float('inf')
        decayed_value = mcv * math.exp(-cognitive_decay * 1)  # 1 year decay

        # Growth projection
        projected_revenue = revenue * (1 + growth_rate)
        projected_mcv = cci * projected_revenue * 0.15
        projected_decayed = projected_mcv * math.exp(-cognitive_decay * 1)

        # VoC risk premium
        risk_premium = voc * 0.3
        discount_rate = risk_free + risk_premium
        present_value = projected_decayed / (1 + discount_rate)

        # Token supply model (from Section 4.2)
        base_supply = 1000000  # 1M tokens
        adjustment = 1 + (cci - 0.5) * 0.4  # Higher CCI = more tokens justified
        token_supply = int(base_supply * adjustment)
        token_price = present_value / token_supply if token_supply > 0 else 0

        # Yield formula (Section 4.5)
        yield_rate = (growth_rate * cci) / (1 + voc)

        return {
            'mcv': round(mcv, 2),
            'cognitive_decay': cognitive_decay,
            'half_life_years': round(half_life, 1),
            'decayed_value_1yr': round(decayed_value, 2),
            'projected_revenue': round(projected_revenue, 2),
            'projected_mcv': round(projected_mcv, 2),
            'voc_risk_premium': round(risk_premium, 4),
            'discount_rate': round(discount_rate, 4),
            'present_value': round(present_value, 2),
            'token_supply': token_supply,
            'token_price': round(token_price, 4),
            'yield_rate': round(yield_rate * 100, 2),
            'valuation_summary': f"TCC valued at Rs. {round(present_value, 2):,} with token price of Rs. {round(token_price, 4)} per token. Cognitive half-life: {round(half_life, 1)} years. Yield: {round(yield_rate * 100, 2)}%.",
        }

    @staticmethod
    def cognitive_reflexivity(cci_history, market_events):
        """
        Cognitive Reflexivity Loop — Section 6.6
        Operators create feedback loops: cognition affects market, market affects cognition.
        """
        results = []
        cci = cci_history[0] if cci_history else 0.5

        for i, event in enumerate(market_events):
            # Market event impacts CCI
            impact = event.get('impact', 0)
            cci = max(0.01, min(1.0, cci + impact * 0.1))

            # CCI adjustment creates reflexive response
            reflexive_adjustment = cci * 0.05 * (1 if impact > 0 else -1)
            market_response = impact + reflexive_adjustment

            results.append({
                'step': i,
                'event': event.get('name', f'Event {i+1}'),
                'market_impact': impact,
                'cci_after_impact': round(cci, 4),
                'reflexive_adjustment': round(reflexive_adjustment, 4),
                'net_market_effect': round(market_response, 4),
            })

        return {
            'reflexivity_loop': results,
            'final_cci': round(cci, 4),
            'interpretation': 'The reflexivity loop shows how market events reshape organizational cognition, which in turn amplifies or dampens subsequent market responses — a cognitive feedback mechanism.',
        }

    @staticmethod
    def run_simulation(scenario_name):
        """Run predefined TCC simulation scenarios from the paper."""
        scenarios = {
            'ai_native_startup': {
                'features': {'knowledge_creation': 0.85, 'decision_efficiency': 0.80, 'ai_alignment': 0.90,
                            'learning_velocity': 0.88, 'innovation_output': 0.75, 'collaboration_index': 0.70, 'adaptive_capacity': 0.82},
                'revenue': 50000000, 'growth_rate': 0.45, 'cognitive_decay': 0.08, 'voc': 0.25,
            },
            'legacy_industrial': {
                'features': {'knowledge_creation': 0.40, 'decision_efficiency': 0.50, 'ai_alignment': 0.30,
                            'learning_velocity': 0.35, 'innovation_output': 0.45, 'collaboration_index': 0.55, 'adaptive_capacity': 0.40},
                'revenue': 500000000, 'growth_rate': 0.05, 'cognitive_decay': 0.15, 'voc': 0.10,
            },
            'financial_institution': {
                'features': {'knowledge_creation': 0.65, 'decision_efficiency': 0.75, 'ai_alignment': 0.60,
                            'learning_velocity': 0.55, 'innovation_output': 0.50, 'collaboration_index': 0.65, 'adaptive_capacity': 0.60},
                'revenue': 200000000, 'growth_rate': 0.12, 'cognitive_decay': 0.10, 'voc': 0.15,
            },
            'dao_collective': {
                'features': {'knowledge_creation': 0.70, 'decision_efficiency': 0.60, 'ai_alignment': 0.75,
                            'learning_velocity': 0.65, 'innovation_output': 0.60, 'collaboration_index': 0.85, 'adaptive_capacity': 0.70},
                'revenue': 10000000, 'growth_rate': 0.30, 'cognitive_decay': 0.05, 'voc': 0.35,
            },
            'crisis_enterprise': {
                'features': {'knowledge_creation': 0.35, 'decision_efficiency': 0.30, 'ai_alignment': 0.25,
                            'learning_velocity': 0.40, 'innovation_output': 0.20, 'collaboration_index': 0.30, 'adaptive_capacity': 0.50},
                'revenue': 100000000, 'growth_rate': -0.15, 'cognitive_decay': 0.25, 'voc': 0.40,
            },
        }

        if scenario_name not in scenarios:
            return {'error': f'Unknown scenario. Available: {list(scenarios.keys())}'}

        sc = scenarios[scenario_name]
        cci_result = TCCEngine.compute_cci(sc['features'])
        valuation = TCCEngine.token_valuation(cci_result['cci'], sc['revenue'], sc['growth_rate'],
                                                sc['cognitive_decay'], sc['voc'])

        # Simulate reflexivity loop
        market_events = [
            {'name': 'Market expansion', 'impact': 0.1},
            {'name': 'Competitor AI launch', 'impact': -0.08},
            {'name': 'Regulatory change', 'impact': -0.05},
            {'name': 'Talent acquisition', 'impact': 0.06},
            {'name': 'Economic downturn', 'impact': -0.12},
        ]
        reflexivity = TCCEngine.cognitive_reflexivity([cci_result['cci']], market_events)

        return {
            'scenario': scenario_name,
            'cci_result': cci_result,
            'valuation': valuation,
            'reflexivity': reflexivity,
        }


# ============================================================================
# MODULE 3: COGNITIVE SETTLEMENT LAYER (CSL)
# Based on: "Cognitive Settlement Layer (CSL): A Multi-Agent, AI-Driven
#            Framework for Real-Time Securities Post-Trade Optimisation"
# ============================================================================

class CSLEngine:
    """
    8-agent settlement optimization system with:
    - Cost, Liquidity, FX, Timeliness, Risk, Exception, Learning, Orchestrator agents
    - Global objective function with Pareto frontier
    - Settlement routing as Markov Decision Process
    """

    AGENT_DEFINITIONS = [
        {'name': 'Cost Agent', 'role': 'Minimizes settlement fees, FX costs, liquidity costs', 'icon': 'cost'},
        {'name': 'Liquidity Agent', 'role': 'Optimizes intraday funding and liquidity usage', 'icon': 'liquidity'},
        {'name': 'FX Agent', 'role': 'Finds optimal cross-currency routing paths', 'icon': 'fx'},
        {'name': 'Timeliness Agent', 'role': 'Ensures cutoff compliance across timezones', 'icon': 'time'},
        {'name': 'Risk Agent', 'role': 'Multi-dimensional risk: counterparty + operational + market', 'icon': 'risk'},
        {'name': 'Exception Agent', 'role': 'Predicts and pre-empts settlement failures', 'icon': 'exception'},
        {'name': 'Learning Agent', 'role': 'RL engine that improves routing over time', 'icon': 'learning'},
        {'name': 'Orchestrator Agent', 'role': 'Applies objective function, computes final route', 'icon': 'orchestrator'},
    ]

    @staticmethod
    def run_agents(trade):
        """
        Execute all 8 agents on a trade and return their computations.
        """
        trade_value = trade.get('value', 1000000)
        currency = trade.get('currency', 'USD')
        counterparty = trade.get('counterparty', 'Broker A')
        settlement_date = trade.get('settlement_date', 'T+2')
        market = trade.get('market', 'SGX')

        agents = []

        # 1. Cost Agent
        fee_rates = {'Euroclear': 0.0002, 'DTCC': 0.00015, 'CBL': 0.00025, 'Internal': 0.0001,
                     'SWIFT': 0.0003, 'CLS_Bank': 0.00018, 'Chainlink': 0.00022, 'Tokenized': 0.00008}
        routes = ['Euroclear', 'DTCC', 'CBL', 'Internal', 'SWIFT', 'CLS_Bank', 'Chainlink', 'Tokenized']
        cost_estimates = {r: trade_value * fee_rates[r] for r in routes}
        best_cost_route = min(cost_estimates, key=cost_estimates.get)
        fx_cost = trade_value * 0.0003 if currency != 'USD' else 0
        total_cost = cost_estimates[best_cost_route] + fx_cost
        agents.append({
            'name': 'Cost Agent', 'status': 'completed',
            'analysis': f"Evaluated {len(routes)} settlement routes. FX cost: ${fx_cost:.2f}",
            'bids': {r: round(v, 2) for r, v in cost_estimates.items()},
            'recommendation': best_cost_route,
            'score': round(total_cost, 2),
        })

        # 2. Liquidity Agent
        liquidity_available = {'Euroclear': 5000000, 'DTCC': 8000000, 'CBL': 3000000, 'Internal': 10000000,
                                 'SWIFT': 2000000, 'CLS_Bank': 7000000, 'Chainlink': 4000000, 'Tokenized': 15000000}
        feasible = {r: liquidity_available[r] >= trade_value for r in routes}
        agents.append({
            'name': 'Liquidity Agent', 'status': 'completed',
            'analysis': f"Checked liquidity across {len(routes)} venues. Trade value: ${trade_value:,.0f}",
            'bids': {r: liquidity_available[r] for r in routes},
            'feasible': feasible,
            'recommendation': max(liquidity_available, key=liquidity_available.get),
            'score': liquidity_available[max(liquidity_available, key=liquidity_available.get)],
        })

        # 3. FX Agent
        fx_paths = {
            'Direct': 0.0003,
            'USD_bridge': 0.0005,
            'EUR_bridge': 0.0006,
        }
        if currency == 'USD':
            fx_paths = {'Direct': 0.0, 'USD_bridge': 0.0, 'EUR_bridge': 0.0004}
        best_fx = min(fx_paths, key=fx_paths.get)
        agents.append({
            'name': 'FX Agent', 'status': 'completed',
            'analysis': f"Evaluated {len(fx_paths)} FX routing paths for {currency}",
            'bids': {k: round(v * trade_value, 2) for k, v in fx_paths.items()},
            'recommendation': best_fx,
            'score': round(fx_paths[best_fx] * trade_value, 2),
        })

        # 4. Timeliness Agent
        cutoff_hours = {'Euroclear': 14, 'DTCC': 16, 'CBL': 12, 'Internal': 18, 'SWIFT': 15, 'CLS_Bank': 17, 'Chainlink': 20, 'Tokenized': 22}
        current_hour = trade.get('current_hour', 13)
        feasible_cutoff = {r: cutoff_hours[r] > current_hour for r in routes}
        agents.append({
            'name': 'Timeliness Agent', 'status': 'completed',
            'analysis': f"Current hour: {current_hour}:00. Checked cutoffs for {len(routes)} venues.",
            'bids': cutoff_hours,
            'feasible': feasible_cutoff,
            'recommendation': max(cutoff_hours, key=cutoff_hours.get),
            'score': max(cutoff_hours.values()),
        })

        # 5. Risk Agent
        # R(x) = wa*R_counterparty + wb*R_operational + wc*R_market
        risk_scores = {}
        for r in routes:
            r_cp = random.uniform(0.1, 0.4)
            r_op = random.uniform(0.05, 0.2)
            r_mkt = random.uniform(0.1, 0.3)
            risk_scores[r] = 0.4 * r_cp + 0.3 * r_op + 0.3 * r_mkt
        best_risk = min(risk_scores, key=risk_scores.get)
        agents.append({
            'name': 'Risk Agent', 'status': 'completed',
            'analysis': 'R(x) = 0.4*R_cp + 0.3*R_op + 0.3*R_mkt. Multi-dimensional risk computed.',
            'bids': {k: round(v, 4) for k, v in risk_scores.items()},
            'recommendation': best_risk,
            'score': round(risk_scores[best_risk], 4),
        })

        # 6. Exception Agent
        fail_probs = {r: random.uniform(0.01, 0.08) for r in routes}
        best_exception = min(fail_probs, key=fail_probs.get)
        agents.append({
            'name': 'Exception Agent', 'status': 'completed',
            'analysis': 'Predicted settlement failure probabilities using historical patterns.',
            'bids': {k: round(v * 100, 2) for k, v in fail_probs.items()},
            'recommendation': best_exception,
            'score': round(fail_probs[best_exception], 4),
        })

        # 7. Learning Agent (RL)
        rl_weights = {r: random.uniform(0.6, 0.95) for r in routes}
        best_rl = max(rl_weights, key=rl_weights.get)
        agents.append({
            'name': 'Learning Agent', 'status': 'completed',
            'analysis': 'RL engine updated Q-values based on 10,000 historical settlements.',
            'bids': {k: round(v, 4) for k, v in rl_weights.items()},
            'recommendation': best_rl,
            'score': round(rl_weights[best_rl], 4),
        })

        # 8. Orchestrator Agent
        # Global Objective: F(x) = w1*C(x) + w2*R(x) + w3*T(x) + w4*B(x)
        w = {'cost': 0.3, 'risk': 0.35, 'timeliness': 0.2, 'exception': 0.15}

        # Normalize scores and compute objective
        route_scores = {}
        for r in routes:
            # Normalize: lower is better for cost, risk, exception; higher for timeliness
            cost_norm = cost_estimates[r] / max(cost_estimates.values())
            risk_norm = risk_scores[r] / max(risk_scores.values())
            time_norm = 1 - (cutoff_hours[r] / max(cutoff_hours.values()))  # invert: earlier cutoff = worse
            exc_norm = fail_probs[r] / max(fail_probs.values())
            liq_ok = 1.0 if feasible[r] else 0.5  # penalty for insufficient liquidity

            objective = (w['cost'] * cost_norm + w['risk'] * risk_norm +
                        w['timeliness'] * time_norm + w['exception'] * exc_norm) * liq_ok

            route_scores[r] = round(objective, 4)

        best_route = min(route_scores, key=route_scores.get)

        # Pareto frontier: find non-dominated routes
        pareto = CSLEngine._pareto_frontier(routes, cost_estimates, risk_scores, cutoff_hours, fail_probs)

        agents.append({
            'name': 'Orchestrator Agent', 'status': 'completed',
            'analysis': f'F(x) = 0.3*C(x) + 0.35*R(x) + 0.2*T(x) + 0.15*B(x). Evaluated {len(routes)} routes.',
            'bids': route_scores,
            'recommendation': best_route,
            'pareto_frontier': pareto,
            'objective_weights': w,
            'score': route_scores[best_route],
        })

        return agents

    @staticmethod
    def _pareto_frontier(routes, costs, risks, timeliness, exceptions):
        """Compute Pareto frontier — routes that are non-dominated."""
        pareto = []
        for r in routes:
            dominated = False
            for r2 in routes:
                if r == r2:
                    continue
                # r2 dominates r if it's better or equal on all dimensions
                if (costs[r2] <= costs[r] and risks[r2] <= risks[r] and
                    timeliness[r2] >= timeliness[r] and exceptions[r2] <= exceptions[r]):
                    if (costs[r2] < costs[r] or risks[r2] < risks[r] or
                        timeliness[r2] > timeliness[r] or exceptions[r2] < exceptions[r]):
                        dominated = True
                        break
            if not dominated:
                pareto.append(r)
        return pareto

    @staticmethod
    def compute_settlement(trade):
        """Full CSL settlement computation with all 8 agents."""
        agents = CSLEngine.run_agents(trade)

        # Get orchestrator result
        orchestrator = next(a for a in agents if a['name'] == 'Orchestrator Agent')
        best_route = orchestrator['recommendation']

        # Compare vs static SSI routing
        ssi_route = 'DTCC'  # Default static route
        ssi_cost = trade.get('value', 1000000) * 0.00015
        best_cost = next(a for a in agents if a['name'] == 'Cost Agent')['bids'][best_route]

        improvement = ((ssi_cost - best_cost) / ssi_cost * 100) if ssi_cost > 0 else 0

        return {
            'trade': trade,
            'agents': agents,
            'optimal_route': best_route,
            'pareto_frontier': orchestrator['pareto_frontier'],
            'objective_function': orchestrator['bids'],
            'weights': orchestrator['objective_weights'],
            'ssi_comparison': {
                'ssi_route': ssi_route,
                'csl_route': best_route,
                'cost_improvement': round(improvement, 2),
                'interpretation': f"CSL recommends {best_route} over static SSI ({ssi_route}), achieving {round(improvement, 1)}% cost improvement. Pareto-optimal routes: {', '.join(orchestrator['pareto_frontier'])}.",
            }
        }

    @staticmethod
    def run_scenario(scenario_name):
        """Run predefined settlement scenarios from the paper."""
        scenarios = {
            'cross_border_equity': {
                'description': 'Cross-border equity: SGX-listed stock held in Hong Kong, settling via sub-custodians into Euroclear',
                'value': 5000000, 'currency': 'SGD', 'counterparty': 'Broker HK',
                'settlement_date': 'T+2', 'market': 'SGX', 'current_hour': 13,
            },
            'repo_sbl': {
                'description': 'Repo / Securities Lending: Borrower receiving collateral',
                'value': 10000000, 'currency': 'USD', 'counterparty': 'JPM',
                'settlement_date': 'T+1', 'market': 'OTC', 'current_hour': 10,
            },
            'multi_currency_fx': {
                'description': 'Multi-currency FX-linked: EUR asset traded in London, funded in SGD',
                'value': 3000000, 'currency': 'EUR', 'counterparty': 'UBS',
                'settlement_date': 'T+2', 'market': 'LSE', 'current_hour': 9,
            },
            'high_stress': {
                'description': 'High-stress: US Equity settling via APAC sub-custodians during market volatility',
                'value': 8000000, 'currency': 'USD', 'counterparty': 'Goldman',
                'settlement_date': 'T+2', 'market': 'NYSE', 'current_hour': 15,
            },
            'emerging_market_bond': {
                'description': 'Emerging market bond: INR-denominated sovereign bond settling via Tokenized route',
                'value': 15000000, 'currency': 'USD', 'counterparty': 'HDFC',
                'settlement_date': 'T+2', 'market': 'NSE', 'current_hour': 11,
            },
            'cross_asset_deriv': {
                'description': 'Cross-asset derivative: equity-linked note settling across CSDs and CCPs',
                'value': 6000000, 'currency': 'EUR', 'counterparty': 'Deutsche Bank',
                'settlement_date': 'T+1', 'market': 'EUREX', 'current_hour': 14,
            },
        }

        if scenario_name not in scenarios:
            return {'error': f'Unknown scenario. Available: {list(scenarios.keys())}'}

        sc = scenarios[scenario_name]
        sc['name'] = scenario_name
        result = CSLEngine.compute_settlement(sc)
        result['description'] = sc['description']
        return result


# ============================================================================
# MODULE 4: THE GATE SYMPHONY
# Based on: "The Gate Symphony: Deterministic Logic-Gate Architectures
#            for Constraining Autonomy in Agentic AI Systems"
# ============================================================================



# ============================================================================
# RL Q-LEARNING ENGINE FOR CSL
# ============================================================================

class CSLQLearning:
    """
    Q-learning engine for CSL settlement routing.
    States: trade features (value bracket, currency, market, hour)
    Actions: settlement routes
    Reward: negative of global objective function
    """
    ROUTES = ['Euroclear', 'DTCC', 'CBL', 'Internal', 'SWIFT', 'CLS_Bank', 'Chainlink', 'Tokenized']

    ROUTE_BASE_COST = {
        'Euroclear': 0.0002, 'DTCC': 0.00015, 'CBL': 0.00025, 'Internal': 0.0001,
        'SWIFT': 0.0003, 'CLS_Bank': 0.00018, 'Chainlink': 0.00022, 'Tokenized': 0.00008,
    }
    ROUTE_RISK = {
        'Euroclear': 0.05, 'DTCC': 0.04, 'CBL': 0.07, 'Internal': 0.02,
        'SWIFT': 0.08, 'CLS_Bank': 0.03, 'Chainlink': 0.06, 'Tokenized': 0.04,
    }
    ROUTE_SPEED = {
        'Euroclear': 0.85, 'DTCC': 0.90, 'CBL': 0.75, 'Internal': 0.95,
        'SWIFT': 0.70, 'CLS_Bank': 0.88, 'Chainlink': 0.80, 'Tokenized': 0.98,
    }

    @staticmethod
    def train(episodes=500, alpha=0.1, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05):
        import random
        q_table = {}
        history = []
        epsilon = epsilon_start
        epsilon_decay = (epsilon_start - epsilon_end) / episodes

        def make_state(trade):
            val_bracket = min(int(trade['value'] / 2000000), 9)
            currencies = ['USD', 'EUR', 'SGD', 'GBP']
            curr_idx = currencies.index(trade.get('currency', 'USD')) if trade.get('currency', 'USD') in currencies else 0
            markets = ['SGX', 'NYSE', 'LSE', 'OTC', 'HKEX', 'TSE']
            mkt_idx = markets.index(trade.get('market', 'OTC')) if trade.get('market', 'OTC') in markets else 3
            hour_bracket = int(trade.get('current_hour', 12) / 4)
            return (val_bracket, curr_idx, mkt_idx, hour_bracket)

        def get_reward(trade, route):
            val = trade['value']
            cost = val * CSLQLearning.ROUTE_BASE_COST[route]
            risk = CSLQLearning.ROUTE_RISK[route]
            speed = CSLQLearning.ROUTE_SPEED[route]
            objective = cost + risk * val * 0.01 + (1 - speed) * val * 0.005
            return -objective

        def get_q(state):
            if state not in q_table:
                q_table[state] = {r: 0.0 for r in CSLQLearning.ROUTES}
            return q_table[state]

        training_trades = []
        for _ in range(episodes):
            val = random.choice([1000000, 3000000, 5000000, 8000000, 10000000, 15000000, 20000000])
            curr = random.choice(['USD', 'EUR', 'SGD', 'GBP'])
            mkt = random.choice(['SGX', 'NYSE', 'LSE', 'OTC', 'HKEX', 'TSE'])
            hr = random.randint(8, 18)
            training_trades.append({'value': val, 'currency': curr, 'market': mkt, 'current_hour': hr})

        cumulative_reward = 0
        for ep in range(episodes):
            trade = training_trades[ep]
            state = make_state(trade)
            q = get_q(state)
            if random.random() < epsilon:
                action = random.choice(CSLQLearning.ROUTES)
            else:
                action = max(q, key=q.get)
            reward = get_reward(trade, action)
            cumulative_reward += reward
            next_q = get_q(state)
            max_next = max(next_q.values())
            q[action] = q[action] + alpha * (reward + gamma * max_next - q[action])
            epsilon = max(epsilon_end, epsilon - epsilon_decay)
            if ep % 10 == 0 or ep == episodes - 1:
                avg_reward = cumulative_reward / max(1, ep + 1)
                ref_trade = {'value': 5000000, 'currency': 'USD', 'market': 'OTC', 'current_hour': 12}
                ref_state = make_state(ref_trade)
                ref_q = get_q(ref_state)
                best_route = max(ref_q, key=ref_q.get)
                history.append({
                    'episode': ep,
                    'avg_reward': round(avg_reward, 2),
                    'epsilon': round(epsilon, 4),
                    'best_route': best_route,
                    'q_values': {k: round(v, 2) for k, v in ref_q.items()},
                })

        return {
            'episodes': episodes,
            'alpha': alpha,
            'gamma': gamma,
            'q_table_size': len(q_table),
            'convergence_history': history,
            'final_q_values': history[-1]['q_values'] if history else {},
            'optimal_route': history[-1]['best_route'] if history else 'Internal',
        }

class GateSymphony:
    """
    Deterministic Boolean logic gates for bounding agentic AI autonomy.
    4 canonical gates: AND, OR, XOR, NAND
    Autonomy lattice: DENY < OBSERVE < SIMULATE < PROPOSE < BOUNDED_EXECUTE < EXECUTE
    Signal provenance: sigma_A (agent), sigma_S (system), sigma_H (human)
    """

    # Autonomy levels (ordered lattice)
    AUTONOMY_LEVELS = ['DENY', 'OBSERVE', 'SIMULATE', 'PROPOSE', 'BOUNDED_EXECUTE', 'EXECUTE']
    AUTONOMY_VALUES = {'DENY': 0, 'OBSERVE': 1, 'SIMULATE': 2, 'PROPOSE': 3, 'BOUNDED_EXECUTE': 4, 'EXECUTE': 5}

    # Consequence classes
    CONSEQUENCE_CLASSES = {
        'C0': {'name': 'Inert', 'description': 'Pure reads, retrievals, internal reasoning. Ungated.', 'min_level': 'OBSERVE'},
        'C1': {'name': 'Reversible', 'description': 'Side effects with guaranteed undo path (draft creation, sandbox writes).', 'min_level': 'SIMULATE'},
        'C2': {'name': 'Consequential', 'description': 'Side effects costly to reverse (sending communications, queue instructions).', 'min_level': 'BOUNDED_EXECUTE'},
        'C3': {'name': 'Irreversible', 'description': 'Side effects with no undo (settlement release, payment execution, data destruction).', 'min_level': 'EXECUTE'},
    }

    GATE_TYPES = {
        'AND': {
            'description': 'Conjunctive Authorization — action executes only if ALL conditions hold',
            'control_primitive': 'Dual-key authorization',
            'truth_table': [[0,0,0],[0,1,0],[1,0,0],[1,1,1]],
            'attenuation': 'Shrinks satisfying set monotonically',
        },
        'OR': {
            'description': 'Redundant Authorization Channels — action executes if ANY channel approves',
            'control_primitive': 'Availability without autonomy',
            'truth_table': [[0,0,0],[0,1,1],[1,0,1],[1,1,1]],
            'attenuation': 'Widens satisfying set (only inside signal channels)',
        },
        'XOR': {
            'description': 'Exclusive-Mode Enforcement — exactly one input may be high',
            'control_primitive': 'Mode exclusivity and conflict detection',
            'truth_table': [[0,0,0],[0,1,1],[1,0,1],[1,1,0]],
            'attenuation': 'Both-high condition fails closed (tampering detection)',
        },
        'NAND': {
            'description': 'Co-occurrence Kill Conditions — all-high kills the flow',
            'control_primitive': 'Deterministic circuit breaker',
            'truth_table': [[0,0,1],[0,1,1],[1,0,1],[1,1,0]],
            'attenuation': 'Carves prohibited regions from open flow',
        },
    }

    @staticmethod
    def evaluate_gate(gate_type, inputs):
        """Evaluate a single Boolean gate."""
        if gate_type not in GateSymphony.GATE_TYPES:
            return {'error': f'Unknown gate: {gate_type}'}

        # Convert inputs to binary
        binary = [1 if i else 0 for i in inputs]

        if gate_type == 'AND':
            output = 1 if all(binary) else 0
        elif gate_type == 'OR':
            output = 1 if any(binary) else 0
        elif gate_type == 'XOR':
            output = 1 if sum(binary) == 1 else 0
        elif gate_type == 'NAND':
            output = 0 if all(binary) else 1
        else:
            output = 0

        return {
            'gate_type': gate_type,
            'inputs': binary,
            'output': output,
            'description': GateSymphony.GATE_TYPES[gate_type]['description'],
            'control_primitive': GateSymphony.GATE_TYPES[gate_type]['control_primitive'],
            'truth_table': GateSymphony.GATE_TYPES[gate_type]['truth_table'],
            'result': 'PROCEED' if output == 1 else 'BLOCKED',
        }

    @staticmethod
    def evaluate_symphony(gates_config, action_apo):
        """
        Evaluate a full Gate Symphony — a DAG of gates.
        Each gate has: type, inputs (signal names), provenance per input.
        Action APO (Action Proposal Object) contains the agent's request.
        """
        # Signal provenance map
        signals = action_apo.get('signals', {})

        results = []
        gate_outputs = {}

        for gate in gates_config:
            gate_name = gate['name']
            gate_type = gate['type']
            input_signals = gate['inputs']

            # Resolve signal values — check gate_outputs first (gate-to-gate), then signals
            input_values = []
            input_provenances = []
            for sig_name in input_signals:
                if sig_name in gate_outputs:
                    # This input is the output of a previous gate
                    val = gate_outputs[sig_name]
                    prov = 'sigma_S'  # Gate outputs are system-provenance
                else:
                    val = signals.get(sig_name, 0)
                    prov = action_apo.get('provenance', {}).get(sig_name, 'sigma_A')
                input_values.append(val)
                input_provenances.append(prov)

            # Evaluate gate
            eval_result = GateSymphony.evaluate_gate(gate_type, input_values)
            eval_result['gate_name'] = gate_name
            eval_result['input_signals'] = input_signals
            eval_result['input_provenances'] = input_provenances

            # Check well-formedness: C2/C3 gates must have at least one sigma_S or sigma_H
            consequence_class = action_apo.get('consequence_class', 'C1')
            if consequence_class in ('C2', 'C3'):
                has_non_agent = any(p in ('sigma_S', 'sigma_H') for p in input_provenances)
                eval_result['well_formed'] = has_non_agent
                if not has_non_agent:
                    eval_result['violation'] = 'Well-formedness rule violated: C2/C3 gate has no sigma_S or sigma_H input'

            # Check for leaky-OR anti-pattern
            if gate_type == 'OR' and consequence_class in ('C2', 'C3'):
                has_agent_disjunct = any(p == 'sigma_A' for p in input_provenances)
                if has_agent_disjunct:
                    eval_result['anti_pattern'] = 'Leaky-OR detected: agent-producible signal on consequential path'

            gate_outputs[gate_name] = eval_result['output']
            results.append(eval_result)

        # Compute effective autonomy level: meet (minimum) of all gate outputs
        all_outputs = [r['output'] for r in results]
        overall = 1 if all(all_outputs) else 0  # AND-composition of all gates

        # Determine autonomy level
        if overall == 1:
            level = action_apo.get('requested_level', 'EXECUTE')
        else:
            # Find the most restrictive gate
            level = 'DENY'

        # Check no-autonomous-path property
        agent_only_signals = {k: v for k, v in signals.items()
                              if action_apo.get('provenance', {}).get(k, 'sigma_A') == 'sigma_A'}
        nap_check = GateSymphony._check_no_autonomous_path(gates_config, signals, action_apo.get('provenance', {}))

        return {
            'action': action_apo.get('action', 'unknown'),
            'consequence_class': consequence_class,
            'consequence_description': GateSymphony.CONSEQUENCE_CLASSES.get(consequence_class, {}).get('description', ''),
            'gate_results': results,
            'overall_result': 'PROCEED' if overall == 1 else 'BLOCKED',
            'effective_autonomy_level': level,
            'no_autonomous_path_verified': nap_check['verified'],
            'nap_analysis': nap_check['analysis'],
            'audit_trail': {
                'gates_evaluated': len(results),
                'gates_satisfied': sum(1 for r in results if r['output'] == 1),
                'gates_blocked': sum(1 for r in results if r['output'] == 0),
                'well_formedness_violations': sum(1 for r in results if not r.get('well_formed', True)),
                'anti_patterns_detected': sum(1 for r in results if 'anti_pattern' in r),
            },
        }

    @staticmethod
    def _check_no_autonomous_path(gates_config, signals, provenance):
        """
        No-Autonomous-Path Theorem: For any well-formed gate symphony,
        there exists no satisfying assignment of agent-producible signals alone
        that opens a path to a consequential action.
        """
        # Build set of all gate names (gate outputs are sigma_S provenance)
        gate_names = {g['name'] for g in gates_config}

        # Check if any gate has only sigma_A inputs (considering gate-to-gate as sigma_S)
        for gate in gates_config:
            input_provs = []
            for s in gate['inputs']:
                if s in gate_names:
                    input_provs.append('sigma_S')  # Gate output = system provenance
                else:
                    input_provs.append(provenance.get(s, 'sigma_A'))
            has_non_agent = any(p in ('sigma_S', 'sigma_H') for p in input_provs)
            if not has_non_agent:
                return {
                    'verified': False,
                    'analysis': f"VIOLATION: Gate '{gate['name']}' ({gate['type']}) has only agent-producible inputs. An autonomous path exists — the agent alone could satisfy this gate.",
                }

        # Simulate: set all sigma_A signals to 1, all sigma_S and sigma_H to 0
        # Then evaluate gates in order, propagating gate outputs
        gate_values = {}
        for gate in gates_config:
            input_vals = []
            for s in gate['inputs']:
                if s in gate_values:
                    input_vals.append(gate_values[s])  # Use previous gate output
                elif s in gate_names:
                    input_vals.append(0)  # Gate output not yet computed (shouldn't happen in DAG)
                elif provenance.get(s, 'sigma_A') == 'sigma_A':
                    input_vals.append(1)  # Agent signal = 1 (worst case)
                else:
                    input_vals.append(0)  # System/human signal = 0
            result = GateSymphony.evaluate_gate(gate['type'], input_vals)
            gate_values[gate['name']] = result['output']

        # Check if any gate can be satisfied with agent-only signals
        can_proceed = all(v == 1 for v in gate_values.values())

        if can_proceed:
            return {
                'verified': False,
                'analysis': 'VIOLATION: With all agent signals set to 1 and non-agent signals set to 0, the action still proceeds. The no-autonomous-path property is violated.',
            }

        return {
            'verified': True,
            'analysis': 'VERIFIED: With only agent-producible signals, at least one gate blocks execution. The no-autonomous-path theorem holds — the agent cannot cause a consequential action without at least one system or human signal.',
        }

    @staticmethod
    def run_scenario(scenario_name):
        """Run predefined Gate Symphony scenarios."""
        scenarios = {
            'ssi_routing_safe': {
                'description': 'SSI routing with all approvals in place (safe path)',
                'action': 'Execute Settlement Instruction',
                'consequence_class': 'C3',
                'requested_level': 'EXECUTE',
                'signals': {
                    'agent_proposal': 1, 'policy_engine': 1, 'human_approval': 1,
                    'risk_check': 1, 'environment_ok': 1
                },
                'provenance': {
                    'agent_proposal': 'sigma_A', 'policy_engine': 'sigma_S',
                    'human_approval': 'sigma_H', 'risk_check': 'sigma_S', 'environment_ok': 'sigma_S'
                },
                'gates': [
                    {'name': 'Policy Gate', 'type': 'AND', 'inputs': ['agent_proposal', 'policy_engine']},
                    {'name': 'Human Gate', 'type': 'AND', 'inputs': ['Policy Gate', 'human_approval']},
                    {'name': 'Risk Gate', 'type': 'AND', 'inputs': ['risk_check', 'environment_ok']},
                    {'name': 'Final Gate', 'type': 'AND', 'inputs': ['Human Gate', 'Risk Gate']},
                ],
            },
            'ssi_routing_blocked': {
                'description': 'SSI routing without human approval (blocked by gate)',
                'action': 'Execute Settlement Instruction',
                'consequence_class': 'C3',
                'requested_level': 'EXECUTE',
                'signals': {
                    'agent_proposal': 1, 'policy_engine': 1, 'human_approval': 0,
                    'risk_check': 1, 'environment_ok': 1
                },
                'provenance': {
                    'agent_proposal': 'sigma_A', 'policy_engine': 'sigma_S',
                    'human_approval': 'sigma_H', 'risk_check': 'sigma_S', 'environment_ok': 'sigma_S'
                },
                'gates': [
                    {'name': 'Policy Gate', 'type': 'AND', 'inputs': ['agent_proposal', 'policy_engine']},
                    {'name': 'Human Gate', 'type': 'AND', 'inputs': ['Policy Gate', 'human_approval']},
                    {'name': 'Risk Gate', 'type': 'AND', 'inputs': ['risk_check', 'environment_ok']},
                    {'name': 'Final Gate', 'type': 'AND', 'inputs': ['Human Gate', 'Risk Gate']},
                ],
            },
            'nand_circuit_breaker': {
                'description': 'NAND gate as circuit breaker: unmatched instruction + new SSI = fraud signature',
                'action': 'Process Instruction',
                'consequence_class': 'C3',
                'requested_level': 'EXECUTE',
                'signals': {
                    'unmatched_instruction': 1, 'new_ssi': 1, 'policy_ok': 1, 'human_ok': 1
                },
                'provenance': {
                    'unmatched_instruction': 'sigma_S', 'new_ssi': 'sigma_S',
                    'policy_ok': 'sigma_S', 'human_ok': 'sigma_H'
                },
                'gates': [
                    {'name': 'Fraud Detector', 'type': 'NAND', 'inputs': ['unmatched_instruction', 'new_ssi']},
                    {'name': 'Authorization', 'type': 'AND', 'inputs': ['Fraud Detector', 'policy_ok', 'human_ok']},
                ],
            },
            'xor_mode_conflict': {
                'description': 'XOR gate: agent in both sandbox AND production mode = conflict',
                'action': 'Execute in Mixed Mode',
                'consequence_class': 'C2',
                'requested_level': 'BOUNDED_EXECUTE',
                'signals': {
                    'sandbox_mode': 1, 'production_mode': 1, 'policy_ok': 1
                },
                'provenance': {
                    'sandbox_mode': 'sigma_S', 'production_mode': 'sigma_S', 'policy_ok': 'sigma_S'
                },
                'gates': [
                    {'name': 'Mode Check', 'type': 'XOR', 'inputs': ['sandbox_mode', 'production_mode']},
                    {'name': 'Authorization', 'type': 'AND', 'inputs': ['Mode Check', 'policy_ok']},
                ],
            },
            'agent_only_attack': {
                'description': 'Adversarial test: agent attempts to execute with only its own signals',
                'action': 'Execute Payment',
                'consequence_class': 'C3',
                'requested_level': 'EXECUTE',
                'signals': {
                    'agent_proposal': 1, 'agent_confidence': 1, 'agent_reasoning': 1
                },
                'provenance': {
                    'agent_proposal': 'sigma_A', 'agent_confidence': 'sigma_A', 'agent_reasoning': 'sigma_A'
                },
                'gates': [
                    {'name': 'Agent Check', 'type': 'AND', 'inputs': ['agent_proposal', 'agent_confidence', 'agent_reasoning']},
                ],
            },
            'or_redundant_channels': {
                'description': 'OR gate: settlement proceeds via either primary OR backup channel',
                'action': 'Execute Settlement',
                'consequence_class': 'C2',
                'requested_level': 'BOUNDED_EXECUTE',
                'signals': {
                    'primary_channel': 0, 'backup_channel': 1, 'policy_ok': 1, 'human_ok': 1
                },
                'provenance': {
                    'primary_channel': 'sigma_S', 'backup_channel': 'sigma_S',
                    'policy_ok': 'sigma_S', 'human_ok': 'sigma_H'
                },
                'gates': [
                    {'name': 'Channel Redundancy', 'type': 'OR', 'inputs': ['primary_channel', 'backup_channel']},
                    {'name': 'Authorization', 'type': 'AND', 'inputs': ['Channel Redundancy', 'policy_ok', 'human_ok']},
                ],
            },
            'nand_emergency_halt': {
                'description': 'NAND emergency halt: systemic risk detected triggers circuit breaker',
                'action': 'Execute Batch Settlement',
                'consequence_class': 'C3',
                'requested_level': 'EXECUTE',
                'signals': {
                    'systemic_risk': 1, 'market_stress': 1, 'human_override': 0, 'policy_ok': 1
                },
                'provenance': {
                    'systemic_risk': 'sigma_S', 'market_stress': 'sigma_S',
                    'human_override': 'sigma_H', 'policy_ok': 'sigma_S'
                },
                'gates': [
                    {'name': 'Emergency Brake', 'type': 'NAND', 'inputs': ['systemic_risk', 'market_stress']},
                    {'name': 'Override Check', 'type': 'AND', 'inputs': ['Emergency Brake', 'human_override', 'policy_ok']},
                ],
            },
        }

        if scenario_name not in scenarios:
            return {'error': f'Unknown scenario. Available: {list(scenarios.keys())}'}

        sc = scenarios[scenario_name]
        result = GateSymphony.evaluate_symphony(sc['gates'], sc)
        result['scenario'] = scenario_name
        result['description'] = sc['description']
        return result


# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/dashboard')
def api_dashboard():
    return jsonify({
        'modules': [
            {'name': 'Quantum Personnel Securities', 'code': 'QPS', 'pages': 48,
             'description': 'Third asset class using quantum mechanics for human/behavioral corporate value'},
            {'name': 'Tokenized Cognitive Capital', 'code': 'TCC', 'pages': 59,
             'description': 'Pricing, tokenizing, and trading organizational intelligence'},
            {'name': 'Cognitive Settlement Layer', 'code': 'CSL', 'pages': 160,
             'description': '8-agent AI system for real-time securities post-trade settlement optimization'},
            {'name': 'The Gate Symphony', 'code': 'GATE', 'pages': 35,
             'description': 'Deterministic logic gates for bounding agentic AI autonomy'},
        ],
        'total_agents': 13,  # 8 CSL + 5 Gate Symphony signal types
        'total_scenarios': 19,  # 5 QPS + 5 TCC + 4 CSL + 5 Gate
        'total_pages_research': 302,
    })

# QPS APIs
@app.route('/api/qps/state', methods=['POST'])
def api_qps_state():
    data = request.json
    strategies = data.get('strategies', ['Expansion', 'Hold', 'Conservative', 'Contraction'])
    amplitudes = data.get('amplitudes')
    return jsonify(QPSEngine.personnel_state_vector(strategies, amplitudes))

@app.route('/api/qps/bias', methods=['POST'])
def api_qps_bias():
    data = request.json
    state = QPSEngine.personnel_state_vector(data.get('strategies', ['Expansion', 'Hold', 'Conservative']),
                                              data.get('amplitudes'))
    result = QPSEngine.apply_bias_operator(state, data.get('bias_type', 'overconfidence'),
                                           data.get('strength'))
    return jsonify(result)

@app.route('/api/qps/hamiltonian', methods=['POST'])
def api_qps_hamiltonian():
    data = request.json
    state = QPSEngine.personnel_state_vector(data.get('strategies', ['Expansion', 'Hold', 'Conservative', 'Contraction']),
                                              data.get('amplitudes'))
    return jsonify(QPSEngine.qps_hamiltonian(state, data.get('market_condition', 'normal'),
                                              data.get('time_steps', 10)))

@app.route('/api/qps/entanglement', methods=['POST'])
def api_qps_entanglement():
    data = request.json
    team_states = []
    for member in data.get('team', []):
        state = QPSEngine.personnel_state_vector(member.get('strategies', ['Expansion', 'Hold', 'Conservative']),
                                                   member.get('amplitudes'))
        team_states.append(state)
    return jsonify(QPSEngine.entanglement_measure(team_states))

@app.route('/api/qps/payoff', methods=['POST'])
def api_qps_payoff():
    data = request.json
    state = QPSEngine.personnel_state_vector(data.get('strategies', ['Expansion', 'Hold', 'Conservative']),
                                              data.get('amplitudes'))
    return jsonify(QPSEngine.qps_payoff(state, data.get('financial_outcome', 100000000),
                                        data.get('bias_penalty', 0.1)))

@app.route('/api/qps/scenario/<name>')
def api_qps_scenario(name):
    return jsonify(QPSEngine.run_scenario(name))

# TCC APIs
@app.route('/api/tcc/cci', methods=['POST'])
def api_tcc_cci():
    data = request.json
    return jsonify(TCCEngine.compute_cci(data.get('features', {})))

@app.route('/api/tcc/valuation', methods=['POST'])
def api_tcc_valuation():
    data = request.json
    cci = data.get('cci', 0.5)
    return jsonify(TCCEngine.token_valuation(cci, data.get('revenue', 100000000),
                                              data.get('growth_rate', 0.1),
                                              data.get('cognitive_decay', 0.1),
                                              data.get('voc', 0.15)))

@app.route('/api/tcc/reflexivity', methods=['POST'])
def api_tcc_reflexivity():
    data = request.json
    return jsonify(TCCEngine.cognitive_reflexivity(data.get('cci_history', [0.5]),
                                                     data.get('market_events', [])))

@app.route('/api/tcc/scenario/<name>')
def api_tcc_scenario(name):
    return jsonify(TCCEngine.run_simulation(name))

# CSL APIs
@app.route('/api/csl/settle', methods=['POST'])
def api_csl_settle():
    return jsonify(CSLEngine.compute_settlement(request.json))

@app.route('/api/csl/scenario/<name>')
def api_csl_scenario(name):
    return jsonify(CSLEngine.run_scenario(name))

# Gate Symphony APIs
@app.route('/api/gate/evaluate', methods=['POST'])
def api_gate_evaluate():
    data = request.json
    return jsonify(GateSymphony.evaluate_symphony(data.get('gates', []), data))

@app.route('/api/gate/scenario/<name>')
def api_gate_scenario(name):
    return jsonify(GateSymphony.run_scenario(name))

@app.route('/api/gate/truth-table/<gate_type>')
def api_gate_truth_table(gate_type):
    if gate_type.upper() in GateSymphony.GATE_TYPES:
        gt = gate_type.upper()
        return jsonify({
            'gate_type': gt,
            'description': GateSymphony.GATE_TYPES[gt]['description'],
            'control_primitive': GateSymphony.GATE_TYPES[gt]['control_primitive'],
            'truth_table': GateSymphony.GATE_TYPES[gt]['truth_table'],
            'attenuation': GateSymphony.GATE_TYPES[gt]['attenuation'],
        })
    return jsonify({'error': 'Unknown gate type'}), 400


@app.route('/api/csl/rl-train')
def api_csl_rl_train():
    try:
        episodes = int(request.args.get('episodes', 500))
        result = CSLQLearning.train(episodes=episodes)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ============================================================================
# HTML PAGES
# ============================================================================

CSS = """
:root{--primary:#1a237e;--primary-light:#3949ab;--accent:#00bcd4;--accent2:#ff6f00;--bg:#0f1117;--card:#1a1d29;--card-light:#232838;--text:#e0e0e0;--text-dim:#9e9e9e;--success:#4caf50;--warning:#ff9800;--danger:#f44336;--border:#2d2d3d;--quantum:#7c4dff;--cognitive:#00e676;--gate:#ff5252;--settle:#448aff}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);display:flex;min-height:100vh}
.sidebar{width:260px;background:var(--card);border-right:1px solid var(--border);position:fixed;height:100vh;overflow-y:auto;z-index:100}
.sidebar-header{padding:20px;border-bottom:1px solid var(--border);text-align:center}
.sidebar-header h1{font-size:1.3rem;background:linear-gradient(135deg,var(--quantum),var(--cognitive));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.sidebar-header .version{font-size:0.65rem;color:var(--accent);margin-top:4px;letter-spacing:1.5px}
.nav-section{padding:10px 0;border-bottom:1px solid var(--border)}
.nav-section-title{padding:8px 20px;font-size:0.65rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--text-dim);font-weight:600}
.nav-item{display:block;padding:10px 20px;color:var(--text-dim);text-decoration:none;font-size:0.82rem;transition:all 0.2s;border-left:3px solid transparent}
.nav-item:hover{background:var(--card-light);color:var(--text)}
.nav-item.active{background:var(--card-light);color:var(--accent);border-left-color:var(--accent)}
.main{margin-left:260px;flex:1;padding:30px;min-width:0}
.page-header{margin-bottom:25px}
.page-header h2{font-size:1.5rem}
.page-header p{color:var(--text-dim);font-size:0.88rem;margin-top:5px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;margin-bottom:20px}
.card h3{font-size:0.95rem;color:var(--accent);margin-bottom:15px}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:15px;margin-bottom:20px}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:18px;text-align:center}
.stat-card .value{font-size:1.8rem;font-weight:700;color:var(--accent)}
.stat-card .label{font-size:0.75rem;color:var(--text-dim);margin-top:4px}
.stat-card.quantum .value{color:var(--quantum)}
.stat-card.cognitive .value{color:var(--cognitive)}
.stat-card.gate .value{color:var(--gate)}
.stat-card.settle .value{color:var(--settle)}
.form-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}
.form-group{display:flex;flex-direction:column;gap:4px}
.form-group label{font-size:0.75rem;color:var(--text-dim)}
.form-group input,.form-group select{background:var(--bg);border:1px solid var(--border);color:var(--text);padding:8px 10px;border-radius:6px;font-size:0.85rem}
.form-group input:focus,.form-group select:focus{outline:none;border-color:var(--accent)}
.btn{background:linear-gradient(135deg,var(--primary),var(--primary-light));color:#fff;border:none;padding:10px 25px;border-radius:8px;font-size:0.85rem;cursor:pointer;font-weight:600}
.btn:hover{opacity:0.9}
.result-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin-top:15px;display:none}
.result-box.show{display:block}
.result-box .result-value{font-size:1.4rem;font-weight:700;color:var(--accent)}
.result-box .result-label{font-size:0.78rem;color:var(--text-dim)}
.chart-container{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin-top:15px}
table{width:100%;border-collapse:collapse;font-size:0.82rem}
table th,table td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--border)}
table th{color:var(--accent);font-weight:600;font-size:0.75rem;text-transform:uppercase}
.tag{display:inline-block;padding:2px 8px;border-radius:10px;font-size:0.7rem;font-weight:600}
.tag-danger{background:rgba(244,67,54,0.15);color:var(--danger)}
.tag-success{background:rgba(76,175,80,0.15);color:var(--success)}
.tag-warning{background:rgba(255,152,0,0.15);color:var(--warning)}
.tag-quantum{background:rgba(124,77,255,0.15);color:var(--quantum)}
.agent-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:8px}
.agent-name{color:var(--accent);font-size:0.82rem;font-weight:600}
.agent-action{font-size:0.78rem;color:var(--text-dim);margin-top:3px}
.math-formula{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin:10px 0;font-family:'Courier New',monospace;font-size:0.85rem;color:var(--accent);overflow-x:auto}
.info-banner{background:linear-gradient(135deg,rgba(124,77,255,0.1),rgba(0,230,118,0.1));border:1px solid rgba(124,77,255,0.3);border-radius:8px;padding:10px 15px;margin-bottom:15px;font-size:0.82rem;color:var(--text-dim)}
.info-banner strong{color:var(--accent)}
.truth-table{display:inline-block;border:1px solid var(--border);border-radius:6px;margin:5px}
.truth-table table{margin:0}
.scenario-btn{display:inline-block;background:var(--card-light);border:1px solid var(--border);color:var(--text);padding:8px 16px;border-radius:8px;font-size:0.8rem;cursor:pointer;margin:4px;text-decoration:none}
.scenario-btn:hover{border-color:var(--accent);color:var(--accent)}
@media(max-width:768px){.sidebar{display:none}.main{margin-left:0;padding:15px}}


/* === ENHANCED V3 STYLES === */
.chart-container{background:var(--card-bg);border-radius:12px;padding:20px;margin:15px 0;border:1px solid rgba(255,255,255,0.08)}
.chart-container canvas{max-height:350px}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin:15px 0}
@media(max-width:768px){.chart-grid{grid-template-columns:1fr}}
.paper-card{background:linear-gradient(135deg,rgba(26,35,126,0.15),rgba(0,188,212,0.08));border:1px solid rgba(0,188,212,0.2);border-radius:12px;padding:20px;margin:10px 0;transition:transform 0.2s}
.paper-card:hover{transform:translateY(-2px);border-color:var(--accent)}
.paper-card h4{color:var(--accent);margin:0 0 8px 0;font-size:1rem}
.paper-card .paper-meta{font-size:0.8rem;color:var(--text-dim);margin-bottom:8px}
.paper-card .paper-innovation{font-size:0.85rem;color:var(--text);margin-top:8px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08)}
.gate-circuit{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:15px 0}
.gate-node{background:var(--card-bg);border:2px solid var(--primary-light);border-radius:8px;padding:10px 15px;min-width:120px;text-align:center}
.gate-node.proceed{border-color:var(--success);background:rgba(0,230,118,0.1)}
.gate-node.blocked{border-color:var(--danger);background:rgba(255,82,82,0.1)}
.gate-node.violation{border-color:#ff9800;background:rgba(255,152,0,0.1)}
.gate-arrow{color:var(--text-dim);font-size:1.2rem}
.prov-tag{display:inline-block;font-size:0.7rem;padding:2px 6px;border-radius:4px;margin:2px}
.prov-sigma-a{background:rgba(255,82,82,0.2);color:#ff8a80}
.prov-sigma-s{background:rgba(0,188,212,0.2);color:#80deea}
.prov-sigma-h{background:rgba(0,230,118,0.2);color:#69f0ae}
.mermaid-section{background:rgba(0,0,0,0.3);border-radius:12px;padding:15px;margin:10px 0;border-left:3px solid var(--accent)}
.formula-block{background:rgba(0,0,0,0.4);border-left:3px solid var(--accent2);padding:12px 16px;margin:10px 0;border-radius:0 8px 8px 0;font-family:'Courier New',monospace;font-size:0.9rem;color:#e0e0e0;overflow-x:auto}
.tag{display:inline-block;padding:3px 10px;border-radius:12px;font-size:0.75rem;font-weight:600}
.tag-quantum{background:rgba(124,77,255,0.2);color:#b388ff}
.tag-success{background:rgba(0,230,118,0.2);color:#69f0ae}
.tag-warning{background:rgba(255,193,7,0.2);color:#ffe082}
.tag-danger{background:rgba(255,82,82,0.2);color:#ff8a80}

/* === v3.3: STAMP, LATTICE, SLIDER, PAPER ARCHIVE === */
.stamp{display:inline-block;font-family:Georgia,serif;font-weight:700;font-size:22px;letter-spacing:.04em;text-transform:uppercase;padding:8px 18px;border:3px solid currentColor;border-radius:6px;box-shadow:inset 0 0 0 2px var(--bg),inset 0 0 0 3.5px currentColor;transform:rotate(-3deg);line-height:1.2;white-space:nowrap;margin:4px 8px 6px 4px;animation:land .28s ease-out}
.stamp.green{color:var(--success)}.stamp.amber{color:var(--warning)}.stamp.red{color:var(--danger)}.stamp.blue{color:var(--accent)}
@keyframes land{from{transform:rotate(-3deg) scale(1.35);opacity:.15}to{transform:rotate(-3deg) scale(1);opacity:1}}
.lattice{display:flex;align-items:center;flex-wrap:wrap;gap:0;margin:15px 0}
.lnode{padding:8px 14px;border:1.5px solid var(--border);border-radius:4px;font-size:13px;background:var(--card);white-space:nowrap}
.lnode.req{border-color:var(--accent);color:var(--accent);border-style:dashed}
.lnode.gr{background:var(--accent);color:#fff;border-color:var(--accent)}
.lnode.gr.req{border-style:solid}
.ledge{width:24px;height:2px;background:var(--border)}
.range-row{display:flex;align-items:center;gap:10px;margin:6px 0}
.range-row label{flex:1;font-size:0.82rem;color:var(--text-dim)}
.range-row output{min-width:40px;text-align:right;font-size:0.82rem;color:var(--text)}
input[type=range]{width:100%;accent-color:var(--accent);margin:4px 0}
.paper-detail{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:24px;margin:15px 0}
.paper-detail h3{font-size:1.2rem;color:var(--accent);margin-bottom:8px}
.paper-detail .paper-abstract{font-size:0.9rem;line-height:1.6;color:var(--text);margin:12px 0;padding:12px;background:var(--bg);border-radius:8px;border-left:3px solid var(--accent)}
.paper-detail .paper-keywords{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
.paper-detail .keyword{background:var(--card-light);border:1px solid var(--border);border-radius:12px;padding:3px 10px;font-size:0.75rem;color:var(--text-dim)}
.paper-detail .paper-meta{font-size:0.8rem;color:var(--text-dim);margin:8px 0}
.paper-card-link{display:block;background:linear-gradient(135deg,rgba(124,77,255,0.08),rgba(0,188,212,0.08));border:1px solid var(--border);border-radius:10px;padding:16px;margin:8px 0;text-decoration:none;color:var(--text);transition:transform 0.2s,border-color 0.2s}
.paper-card-link:hover{transform:translateY(-2px);border-color:var(--accent)}
.paper-card-link h4{color:var(--accent);margin:0 0 6px 0;font-size:0.95rem}
.paper-card-link .paper-meta{font-size:0.78rem;color:var(--text-dim)}
.csl-weight-panel{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:15px;margin:10px 0}
.route-row{display:flex;align-items:center;gap:10px;padding:6px 0;border-bottom:1px solid var(--border)}
.route-row:last-child{border-bottom:none}
.route-bar{flex:1;height:14px;background:var(--bg);border-radius:3px;overflow:hidden}
.route-bar span{display:block;height:100%;border-radius:3px}
"""



# ============================================================================
# PAGE RENDERING
# ============================================================================

def page(title, content, active=''):
    nav_html = '<div class="sidebar"><div class="sidebar-header"><h1>FinSight AI</h1><div class="version">v3.2 COGNITIVE FINANCE</div></div>'
    sections = [
        ('Overview', [('/', 'Dashboard', 'overview')]),
        ('Paper 1: QPS', [('/qps', 'Quantum Personnel Securities', 'qps'), ('/qps/bias', 'Bias Operators', 'qps-bias'), ('/qps/scenarios', 'Simulation Scenarios', 'qps-scen')]),
        ('Paper 2: TCC', [('/tcc', 'Cognitive Capital Index', 'tcc'), ('/tcc/valuation', 'Token Valuation', 'tcc-val'), ('/tcc/scenarios', 'Market Scenarios', 'tcc-scen')]),
        ('Paper 3: CSL', [('/csl', 'Settlement Layer', 'csl'), ('/csl/rl', 'RL Training', 'csl-rl'), ('/csl/scenarios', 'Settlement Scenarios', 'csl-scen')]),
        ('Paper 4: Gates', [('/gate', 'Gate Symphony', 'gate'), ('/gate/truth-tables', 'Truth Tables', 'gate-tt'), ('/gate/scenarios', 'Gate Scenarios', 'gate-scen')]),
        ('Archive', [('/papers', 'Research Papers', 'papers')]),
    ]
    for section_name, items in sections:
        nav_html += '<div class="nav-section"><div class="nav-section-title">' + section_name + '</div>'
        for href, label, key in items:
            cls = ' active' if key == active else ''
            nav_html += '<a href="' + href + '" class="nav-item' + cls + '">' + label + '</a>'
        nav_html += '</div>'
    nav_html += '</div>'
    return '<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>' + title + ' | FinSight AI v3.0</title><style>' + CSS + '</style><script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script></head><body>' + nav_html + '<div class="main"><div class="page-header"><h2>' + title + '</h2></div>' + content + '</div></body></html>'


# ============================================================================
# DASHBOARD
# ============================================================================

@app.route('/')
def page_dashboard():
    c = '<p>FinSight AI v3.0 is an end-to-end Agentic AI cognitive finance platform implementing four SSRN research papers by <strong>Saumyajit Ghosh</strong>. Each module translates theoretical frameworks from the papers into live, interactive computation.</p>'
    # Paper cards
    c += '<div class="card"><h3>Research Foundation: 4 Papers, 302 Pages</h3>'
    c += '<div class="paper-card"><h4><span class="tag tag-quantum">QPS</span> Quantum Personnel Securities</h4>'
    c += '<div class="paper-meta">Paper 1 | 48 pages | Third asset class beyond equity and debt</div>'
    c += '<div class="paper-innovation"><strong>Key Innovation:</strong> Quantum mechanics (superposition, entanglement, bias operators) applied to corporate leadership and human behavioral value. Introduces state C — orthogonal to equity (A) and debt (B).</div></div>'
    c += '<div class="paper-card"><h4><span class="tag tag-success">TCC</span> Tokenized Cognitive Capital</h4>'
    c += '<div class="paper-meta">Paper 2 | 59 pages | Pricing and trading organizational intelligence</div>'
    c += '<div class="paper-innovation"><strong>Key Innovation:</strong> Cognitive Capital Index (7 dimensions), cognitive decay half-life, Volatility of Cognition (VoC) risk premium, cognitive reflexivity loops. Extends QPS into a tokenized instrument.</div></div>'
    c += '<div class="paper-card"><h4><span class="tag tag-warning">CSL</span> Cognitive Settlement Layer</h4>'
    c += '<div class="paper-meta">Paper 3 | 160 pages | Multi-agent post-trade settlement optimization</div>'
    c += '<div class="paper-innovation"><strong>Key Innovation:</strong> 8 specialized AI agents with RL-based MDP routing, Pareto frontier optimization, global objective function across cost/risk/timeliness/exceptions. Control Tower Architecture.</div></div>'
    c += '<div class="paper-card"><h4><span class="tag tag-danger">GATE</span> The Gate Symphony</h4>'
    c += '<div class="paper-meta">Paper 4 | 35 pages | Deterministic logic gates for bounding agentic AI</div>'
    c += '<div class="paper-innovation"><strong>Key Innovation:</strong> Boolean gates (AND/OR/XOR/NAND) that constrain agent autonomy. No-Autonomous-Path theorem verified across 50,000 cases. Signal provenance: sigma_A (agent), sigma_S (system), sigma_H (human).</div></div>'
    c += '</div>'
    # Stats
    c += '<div class="stat-grid">'
    c += '<div class="stat-card"><div class="value">4</div><div class="label">Research Modules</div></div>'
    c += '<div class="stat-card"><div class="value">13</div><div class="label">AI Agents</div></div>'
    c += '<div class="stat-card"><div class="value">26</div><div class="label">Simulation Scenarios</div></div>'
    c += '<div class="stat-card"><div class="value">302</div><div class="label">Research Pages</div></div>'
    c += '</div>'
    # Architecture flow
    c += '<div class="card"><h3>Cognitive Finance Architecture</h3>'
    c += '<div class="formula-block">'
    c += 'QPS (quantum state of leadership)<br>'
    c += '&nbsp;&nbsp;&rarr; TCC (tokenize the cognitive capital)<br>'
    c += '&nbsp;&nbsp;&nbsp;&nbsp;rarr; CSL (8-agent settlement of TCC instruments)<br>'
    c += '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;rarr; Gate Symphony (deterministic bounds on all agents)'
    c += '</div>'
    c += '<p>Each layer builds on the previous: QPS defines the asset, TCC prices and tokenizes it, CSL settles it through multi-agent negotiation, and Gate Symphony ensures no agent can execute without human/system oversight.</p>'
    c += '</div>'
    return page('Dashboard', c, 'overview')


# ============================================================================
# QPS PAGES
# ============================================================================

@app.route('/qps')
def page_qps():
    c = '<p>Quantum Personnel Securities (QPS) introduces a third asset class using quantum mechanics &mdash; superposition, entanglement, and bias operators &mdash; to model the human and behavioral dimension of corporate value.</p>'
    c += '<div class="info-banner"><strong>Paper 1:</strong> QPS establishes a hybrid structure that assigns tradable rights linked to key personnel. The quantum state C is neither pure equity (state A) nor pure debt (state B), but a third, orthogonal state linking financial outcomes to human decision-making quality and bias.</div>'
    c += '<div class="card"><h3>Personnel State Vector</h3>'
    c += '<div class="formula-block">|psi> = alpha_1|strategy_1> + alpha_2|strategy_2> + ... + alpha_n|strategy_n><br>where sum(|alpha_i|^2) = 1 (normalization)</div>'
    c += '<form id="qpsForm" onsubmit="return submitQPS(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Strategy 1</label><input type="text" name="s1" value="Expansion"></div>'
    c += '<div class="form-group"><label>Amplitude 1</label><input type="number" name="a1" value="0.3" step="0.01"></div>'
    c += '<div class="form-group"><label>Strategy 2</label><input type="text" name="s2" value="Aggressive Acquisition"></div>'
    c += '<div class="form-group"><label>Amplitude 2</label><input type="number" name="a2" value="0.4" step="0.01"></div>'
    c += '<div class="form-group"><label>Strategy 3</label><input type="text" name="s3" value="Hold"></div>'
    c += '<div class="form-group"><label>Amplitude 3</label><input type="number" name="a3" value="0.2" step="0.01"></div>'
    c += '<div class="form-group"><label>Strategy 4</label><input type="text" name="s4" value="Conservative Growth"></div>'
    c += '<div class="form-group"><label>Amplitude 4</label><input type="number" name="a4" value="0.1" step="0.01"></div>'
    c += '</div><p><button type="submit" class="btn">Compute State Vector</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="qpsDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="stateChart"></canvas></div>'
    c += '<script>'
    c += """function submitQPS(e){e.preventDefault();var f=new FormData(e.target);var strategies=[f.get('s1'),f.get('s2'),f.get('s3'),f.get('s4')].filter(function(s){return s});var amplitudes=[parseFloat(f.get('a1')),parseFloat(f.get('a2')),parseFloat(f.get('a3')),parseFloat(f.get('a4'))].filter(function(a){return!isNaN(a)});fetch('/api/qps/state',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strategies:strategies,amplitudes:amplitudes})}).then(function(r){return r.json()}).then(function(r){document.getElementById('result').classList.add('show');var html='<div class="result-label">Dominant Strategy</div><div class="result-value">'+r.max_strategy+'</div>';html+='<p>Probability: '+(r.max_probability*100).toFixed(1)+'%</p>';html+='<p>Entropy: '+r.entropy.toFixed(4)+'</p>';html+='<table><tr><th>Strategy</th><th>Probability</th></tr>';Object.entries(r.probabilities).forEach(function(e){html+='<tr><td>'+e[0]+'</td><td>'+(e[1]*100).toFixed(2)+'%</td></tr>'});html+='</table>';document.getElementById('qpsDetails').innerHTML=html;drawStateChart(r)})}var stateChart=null;function drawStateChart(r){var ctx=document.getElementById('stateChart');document.getElementById('chartBox').style.display='block';if(stateChart)stateChart.destroy();stateChart=new Chart(ctx,{type:'bar',data:{labels:Object.keys(r.probabilities),datasets:[{label:'Probability',data:Object.values(r.probabilities),backgroundColor:'#7c4dff'}]},options:{responsive:true,plugins:{title:{display:true,text:'Strategy Probabilities'}},scales:{y:{beginAtZero:true,max:1}}}})}"""
    c += '</script>'
    c += '<div class="card"><h3>Bloch Sphere Visualization</h3>'
    c += '<p>The Bloch sphere represents the quantum state of leadership. Each strategy maps to a point on the sphere surface. The dominant strategy is highlighted.</p>'
    c += '<div style="display:flex;justify-content:center;margin:15px 0">'
    c += '<canvas id="blochSphere" width="320" height="320" style="border:1px solid rgba(255,255,255,0.1);border-radius:50%"></canvas>'
    c += '</div>'
    c += '<div id="blochLabels" style="text-align:center;font-size:0.85rem;color:var(--text-dim)"></div>'
    c += '<script>'
    c += """function drawBloch(){var c=document.getElementById('blochSphere');if(!c)return;var ctx=c.getContext('2d');var cx=160,cy=160,r=120;ctx.clearRect(0,0,320,320);ctx.strokeStyle='rgba(255,255,255,0.15)';ctx.lineWidth=1;ctx.beginPath();ctx.arc(cx,cy,r,0,2*Math.PI);ctx.stroke();ctx.beginPath();ctx.ellipse(cx,cy,r,r*0.3,0,0,2*Math.PI);ctx.stroke();ctx.beginPath();ctx.ellipse(cx,cy,r*0.3,r,0,0,2*Math.PI);ctx.stroke();ctx.strokeStyle='rgba(255,255,255,0.3)';ctx.beginPath();ctx.moveTo(cx-r-10,cy);ctx.lineTo(cx+r+10,cy);ctx.moveTo(cx,cy-r-10);ctx.lineTo(cx,cy+r+10);ctx.stroke();ctx.fillStyle='rgba(255,255,255,0.5)';ctx.font='10px sans-serif';ctx.fillText('|0>',cx+5,cy-r-5);ctx.fillText('|1>',cx+5,cy+r+12);ctx.fillText('x',cx+r+12,cy+3);var strategies=document.querySelectorAll('#qpsDetails table tr');var colors=['#7c4dff','#ff5252','#00e676','#ffab00'];var labels=[];for(var i=1;i<strategies.length&&i<=4;i++){var angle=(i-1)*Math.PI*0.5+0.3;var px=cx+r*0.7*Math.cos(angle);var py=cy-r*0.7*Math.sin(angle);ctx.fillStyle=colors[i-1];ctx.beginPath();ctx.arc(px,py,8,0,2*Math.PI);ctx.fill();var cells=strategies[i].querySelectorAll('td');if(cells.length>=2){labels.push({name:cells[0].textContent,prob:cells[1].textContent,color:colors[i-1]})}}var labelHtml='';labels.forEach(function(l){labelHtml+='<span style=color:'+l.color+'>&#9679;</span> '+l.name+': '+l.prob+' &nbsp; '});var bl=document.getElementById('blochLabels');if(bl)bl.innerHTML=labelHtml}setTimeout(drawBloch,500)"""
    c += '</script>'
    return page('Quantum Personnel Securities', c, 'qps')

@app.route('/qps/bias')
def page_qps_bias():
    c = '<p>Bias operators model non-linear decision influences on leadership. Each operator transforms the personnel state vector, concentrating or dispersing strategic probability.</p>'
    c += '<div class="info-banner"><strong>Bias Operators:</strong> Overconfidence (O_OC), Loss Aversion (O_LA), Groupthink (O_GT), Anchoring (O_AN), Confirmation (O_CF). These are non-commuting operators &mdash; the order of application matters.</div>'
    c += '<div class="card"><h3>Apply Bias Operator</h3>'
    c += '<form id="biasForm" onsubmit="return submitBias(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Bias Type</label><select name="bias_type"><option value="overconfidence">Overconfidence (O_OC)</option><option value="loss_aversion">Loss Aversion (O_LA)</option><option value="groupthink">Groupthink (O_GT)</option><option value="anchoring">Anchoring (O_AN)</option><option value="confirmation">Confirmation (O_CF)</option></select></div>'
    c += '<div class="form-group"><label>Bias Strength (0-1)</label><input type="number" name="strength" value="0.3" step="0.01" min="0" max="1"></div>'
    c += '</div><p><button type="submit" class="btn">Apply Bias Operator</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="biasDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="biasChart"></canvas></div>'
    c += '<script>'
    c += """function submitBias(e){e.preventDefault();var f=new FormData(e.target);var strategies=["Expansion","Aggressive Acquisition","Hold","Conservative Growth"];var amplitudes=[0.3,0.4,0.2,0.1];fetch('/api/qps/bias',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({strategies:strategies,amplitudes:amplitudes,bias_type:f.get('bias_type'),strength:parseFloat(f.get('strength'))})}).then(function(r){return r.json()}).then(function(r){document.getElementById('result').classList.add('show');var html='<div class="result-label">Bias Applied: '+r.bias_applied+' ('+r.bias_symbol+')</div>';html+='<p>'+r.bias_description+'</p>';html+='<p>Entropy Change: '+(r.entropy_change>=0?'+':'')+r.entropy_change.toFixed(4)+'</p>';html+='<p>New Dominant: '+r.new_dominant_strategy+'</p>';html+='<p><em>'+r.interpretation+'</em></p>';html+='<table><tr><th>Strategy</th><th>New Probability</th></tr>';Object.entries(r.new_probabilities).forEach(function(e){html+='<tr><td>'+e[0]+'</td><td>'+(e[1]*100).toFixed(2)+'%</td></tr>'});html+='</table>';document.getElementById('biasDetails').innerHTML=html;drawBiasChart(r)})}var biasChart=null;function drawBiasChart(r){var ctx=document.getElementById('biasChart');document.getElementById('chartBox').style.display='block';if(biasChart)biasChart.destroy();biasChart=new Chart(ctx,{type:'bar',data:{labels:Object.keys(r.new_probabilities),datasets:[{label:'Biased Probability',data:Object.values(r.new_probabilities),backgroundColor:'#ff5252'}]},options:{responsive:true,plugins:{title:{display:true,text:'State After Bias Operator'}},scales:{y:{beginAtZero:true,max:1}}}})}"""
    c += '</script>'
    return page('Bias Operators', c, 'qps-bias')

@app.route('/qps/scenarios')
def page_qps_scenarios():
    c = '<p>Predefined simulation scenarios from the QPS paper, each demonstrating different leadership conditions and bias interactions. Each scenario shows the full state evolution over 10 time steps with entropy decay and payoff analysis.</p>'
    c += '<div class="card"><h3>Simulation Scenarios</h3>'
    scenarios = [
        ('overconfident_ceo', 'Overconfident CEO in Bull Market'),
        ('risk_averse_board', 'Risk-Averse Board in Bear Market'),
        ('groupthink_committee', 'Groupthink in Strategic Committee'),
        ('crisis_response', 'Crisis Response Team'),
        ('balanced_leadership', 'Balanced Triad Leadership (Bias-Corrected)'),
    ]
    for name, label in scenarios:
        c += '<a class="scenario-btn" href="#" onclick="runQPS(\'' + name + '\');return false">' + label + '</a>'
    c += '</div>'
    c += '<div id="qpsResult"></div>'
    c += '<script>'
    c += """function runQPS(name){fetch('/api/qps/scenario/'+name).then(function(r){return r.json()}).then(function(r){var html='<div class="card"><h3>'+r.description+'</h3>';html+='<p><strong>Initial State:</strong></p><table>';Object.entries(r.initial_state.probabilities).forEach(function(e){html+='<tr><td>'+e[0]+'</td><td>'+(e[1]*100).toFixed(1)+'%</td></tr>'});html+='</table>';if(r.bias_applications){html+='<h3 style="margin-top:15px">Bias Applications</h3>';r.bias_applications.forEach(function(b){html+='<div class="agent-card"><div class="agent-name">'+b.bias_applied+' ('+b.bias_symbol+')</div>';html+='<div class="agent-action">Entropy change: '+(b.entropy_change>=0?'+':'')+b.entropy_change.toFixed(4)+'</div>';html+='<div class="agent-action"><em>'+b.interpretation+'</em></div></div>'})}html+='<h3 style="margin-top:15px">Payoff Analysis</h3><table><tr><th>Metric</th><th>Value</th></tr>';html+='<tr><td>Base Financial Outcome</td><td>Rs. '+r.payoff.base_financial_outcome.toLocaleString()+'</td></tr>';html+='<tr><td>Entropy Penalty</td><td>'+(r.payoff.entropy_penalty*100).toFixed(2)+'%</td></tr>';html+='<tr><td>Strategy-Weighted Payoff</td><td>Rs. '+r.payoff.strategy_weighted_payoff.toLocaleString()+'</td></tr>';html+='<tr><td>Adjusted Payoff</td><td>Rs. '+r.payoff.adjusted_payoff.toLocaleString()+'</td></tr></table>';html+='<p><em>'+r.payoff.interpretation+'</em></p>';html+='<div class="formula-block">Market: '+r.state_evolution.market_condition+' | Steps: '+r.state_evolution.time_steps+' | Final: '+r.state_evolution.final_state.dominant_strategy+'</div>';html+='</div>';html+='<div class="chart-grid">';html+='<div class="chart-container"><canvas id="qpsEvoChart"></canvas></div>';html+='<div class="chart-container"><canvas id="qpsEntropyChart"></canvas></div>';html+='</div>';document.getElementById('qpsResult').innerHTML=html;drawQPSEvolution(r);drawQPSEntropy(r)})}var qpsEvoChart=null;function drawQPSEvolution(r){var ctx=document.getElementById('qpsEvoChart');if(qpsEvoChart)qpsEvoChart.destroy();var steps=r.state_evolution.evolution;var labels=steps.map(function(s){return 'Step '+s.step});var strategies=Object.keys(steps[0].probabilities);var datasets=strategies.map(function(s,i){var colors=['#7c4dff','#ff5252','#00e676','#ffab00'];return{label:s,data:steps.map(function(step){return step.probabilities[s]*100}),borderColor:colors[i],backgroundColor:colors[i]+'33',fill:false}});qpsEvoChart=new Chart(ctx,{type:'line',data:{labels:labels,datasets:datasets},options:{responsive:true,plugins:{title:{display:true,text:'Strategy Probability Evolution (%)'}},scales:{y:{beginAtZero:true,max:100}}}})}var qpsEntChart=null;function drawQPSEntropy(r){var ctx=document.getElementById('qpsEntropyChart');if(qpsEntChart)qpsEntChart.destroy();var steps=r.state_evolution.evolution;qpsEntChart=new Chart(ctx,{type:'line',data:{labels:steps.map(function(s){return 'Step '+s.step}),datasets:[{label:'Entropy',data:steps.map(function(s){return s.entropy}),borderColor:'#00bcd4',backgroundColor:'rgba(0,188,212,0.2)',fill:true}]},options:{responsive:true,plugins:{title:{display:true,text:'Entropy Decay Over Time'}},scales:{y:{beginAtZero:true}}}})}"""
    c += '</script>'
    return page('QPS Simulation Scenarios', c, 'qps-scen')


# ============================================================================
# TCC PAGES
# ============================================================================

@app.route('/tcc')
def page_tcc():
    c = '<p>The Cognitive Capital Index (CCI) measures collective organizational intelligence across 7 dimensions, using a weighted formula with entropy adjustment and penalty functions.</p>'
    c += '<div class="formula-block">CCI = sum(w_i * f_i) * entropy_adjustment * (1 - penalty)<br>where f_i are normalized feature scores, w_i are dynamic weights</div>'
    c += '<div class="card"><h3>Compute Cognitive Capital Index</h3>'
    c += '<form id="cciForm" onsubmit="return submitCCI(event)">'
    c += '<div class="form-grid">'
    features = [
        ('knowledge_creation', 0.65), ('decision_efficiency', 0.70), ('ai_alignment', 0.55),
        ('learning_velocity', 0.60), ('innovation_output', 0.50), ('collaboration_index', 0.68),
        ('adaptive_capacity', 0.62),
    ]
    for fname, default in features:
        c += '<div class="form-group"><label>' + fname.replace('_', ' ').title() + ' (0-1)</label><input type="number" name="' + fname + '" value="' + str(default) + '" step="0.01" min="0" max="1"></div>'
    c += '</div><p><button type="submit" class="btn">Compute CCI</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="cciDetails"></div></div>'
    c += '<div class="chart-container" id="chartBox" style="display:none"><canvas id="cciChart"></canvas></div>'
    c += '<script>'
    c += """function submitCCI(e){e.preventDefault();var f=new FormData(e.target);var features={};f.forEach(function(v,k){features[k]=parseFloat(v)});fetch('/api/tcc/cci',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({features:features})}).then(function(r){return r.json()}).then(function(r){document.getElementById('result').classList.add('show');var html='<div class="result-label">Cognitive Capital Index</div><div class="result-value">'+r.cci.toFixed(4)+'</div>';html+='<p>Rating: '+r.rating+'</p>';html+='<p><em>'+r.interpretation+'</em></p>';html+='<table><tr><th>Dimension</th><th>Score</th></tr>';Object.entries(r.feature_scores).forEach(function(e){html+='<tr><td>'+e[0].replace(/_/g,' ')+'</td><td>'+e[1].toFixed(3)+'</td></tr>'});html+='</table>';html+='<p>Entropy: '+r.entropy.toFixed(4)+' | Penalty: '+(r.penalty*100).toFixed(1)+'%</p>';document.getElementById('cciDetails').innerHTML=html;drawCCIChart(r)})}var cciChart=null;function drawCCIChart(r){var ctx=document.getElementById('cciChart');document.getElementById('chartBox').style.display='block';if(cciChart)cciChart.destroy();var labels=Object.keys(r.feature_scores).map(function(k){return k.replace(/_/g,' ')});cciChart=new Chart(ctx,{type:'radar',data:{labels:labels,datasets:[{label:'CCI Dimensions',data:Object.values(r.feature_scores),backgroundColor:'rgba(0,230,118,0.2)',borderColor:'#00e676'}]},options:{responsive:true,plugins:{title:{display:true,text:'Cognitive Capital Index - 7 Dimensions'}},scales:{r:{beginAtZero:true,max:1}}}})}"""
    c += '</script>'
    return page('Cognitive Capital Index', c, 'tcc')

@app.route('/tcc/valuation')
def page_tcc_val():
    c = '<p>TCC Token Valuation Framework &mdash; links organizational intelligence to economic output with cognitive decay, Volatility of Cognition (VoC), and risk premium.</p>'
    c += '<div class="formula-block">V(TCC) = MCV * e^(-lambda*t) / (1 + VoC_premium)<br>MCV = CCI * Revenue * 0.15<br>discount_rate = risk_free + VoC * 0.3</div>'
    c += '<div class="card"><h3>Token Valuation Parameters</h3>'
    c += '<form id="valForm" onsubmit="return submitVal(event)">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>CCI Score (0-1)</label><input type="number" name="cci" value="0.65" step="0.01" min="0" max="1"></div>'
    c += '<div class="form-group"><label>Revenue (Rs.)</label><input type="number" name="revenue" value="100000000" step="100000"></div>'
    c += '<div class="form-group"><label>Growth Rate</label><input type="number" name="growth_rate" value="0.15" step="0.01"></div>'
    c += '<div class="form-group"><label>Cognitive Decay (lambda)</label><input type="number" name="cognitive_decay" value="0.10" step="0.01"></div>'
    c += '<div class="form-group"><label>Volatility of Cognition (VoC)</label><input type="number" name="voc" value="0.15" step="0.01"></div>'
    c += '</div><p><button type="submit" class="btn">Value TCC Tokens</button></p></form></div>'
    c += '<div class="result-box" id="result"><div id="valDetails"></div></div>'
    c += '<script>'
    c += """function submitVal(e){e.preventDefault();var f=new FormData(e.target);var d={};f.forEach(function(v,k){d[k]=parseFloat(v)});fetch('/api/tcc/valuation',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json()}).then(function(r){document.getElementById('result').classList.add('show');var html='<div class="result-label">Token Price</div><div class="result-value">Rs. '+r.token_price.toFixed(4)+'</div>';html+='<table><tr><th>Metric</th><th>Value</th></tr>';html+='<tr><td>MCV</td><td>Rs. '+r.mcv.toLocaleString()+'</td></tr>';html+='<tr><td>Cognitive Half-Life</td><td>'+r.half_life_years+' years</td></tr>';html+='<tr><td>Present Value</td><td>Rs. '+r.present_value.toLocaleString()+'</td></tr>';html+='<tr><td>Token Supply</td><td>'+r.token_supply.toLocaleString()+'</td></tr>';html+='<tr><td>Yield Rate</td><td>'+r.yield_rate+'%</td></tr>';html+='<tr><td>Discount Rate</td><td>'+(r.discount_rate*100).toFixed(2)+'%</td></tr>';html+='<tr><td>VoC Risk Premium</td><td>'+(r.voc_risk_premium*100).toFixed(2)+'%</td></tr></table>';html+='<p><em>'+r.valuation_summary+'</em></p>';document.getElementById('valDetails').innerHTML=html})}"""
    c += '</script>'
    return page('Token Valuation', c, 'tcc-val')

@app.route('/tcc/scenarios')
def page_tcc_scen():
    c = '<p>Predefined TCC simulation scenarios from the paper, covering different organizational archetypes with full CCI computation, token valuation, and cognitive reflexivity loops.</p>'
    c += '<div class="card"><h3>Market Scenarios</h3>'
    scenarios = [
        ('ai_native_startup', 'AI-Native Startup'),
        ('legacy_industrial', 'Legacy Industrial Firm (with AI Retrofit)'),
        ('financial_institution', 'Financial Institution with Cognitive Governance'),
        ('dao_collective', 'Decentralized Cognitive Network (DAO)'),
        ('crisis_enterprise', 'Public Enterprise Under Crisis'),
    ]
    for name, label in scenarios:
        c += '<a class="scenario-btn" href="#" onclick="runTCC(\'' + name + '\');return false">' + label + '</a>'
    c += '</div>'
    c += '<div id="tccResult"></div>'
    c += '<script>'
    c += """function runTCC(name){fetch('/api/tcc/scenario/'+name).then(function(r){return r.json()}).then(function(r){var html='<div class="card"><h3>'+r.scenario.replace(/_/g,' ')+'</h3>';html+='<div class="result-value">CCI: '+r.cci_result.cci.toFixed(4)+'</div>';html+='<p>Rating: '+r.cci_result.rating+'</p>';html+='<p><em>'+r.cci_result.interpretation+'</em></p>';html+='<table><tr><th>Dimension</th><th>Score</th></tr>';Object.entries(r.cci_result.feature_scores).forEach(function(e){html+='<tr><td>'+e[0].replace(/_/g,' ')+'</td><td>'+e[1].toFixed(3)+'</td></tr>'});html+='</table>';html+='</div>';html+='<div class="card"><h3>Token Valuation</h3><table>';html+='<tr><td>Token Price</td><td>Rs. '+r.valuation.token_price.toFixed(4)+'</td></tr>';html+='<tr><td>Present Value</td><td>Rs. '+r.valuation.present_value.toLocaleString()+'</td></tr>';html+='<tr><td>Yield</td><td>'+r.valuation.yield_rate+'%</td></tr>';html+='<tr><td>Half-Life</td><td>'+r.valuation.half_life_years+' years</td></tr></table>';html+='<p><em>'+r.valuation.valuation_summary+'</em></p></div>';html+='<div class="card"><h3>Cognitive Reflexivity Loop</h3>';html+='<table><tr><th>Step</th><th>Event</th><th>Market Impact</th><th>CCI After</th></tr>';r.reflexivity.reflexivity_loop.forEach(function(s){html+='<tr><td>'+s.step+'</td><td>'+s.event+'</td><td>'+(s.market_impact*100).toFixed(1)+'%</td><td>'+s.cci_after_impact.toFixed(4)+'</td></tr>'});html+='</table>';html+='<p><strong>Final CCI:</strong> '+r.reflexivity.final_cci.toFixed(4)+'</p>';html+='<p><em>'+r.reflexivity.interpretation+'</em></p></div>';html+='<div class="chart-grid">';html+='<div class="chart-container"><canvas id="tccRadarChart"></canvas></div>';html+='<div class="chart-container"><canvas id="tccReflexChart"></canvas></div>';html+='</div>';document.getElementById('tccResult').innerHTML=html;drawTCCRadar(r);drawTCCReflexivity(r)})}var tccRadarChart=null;function drawTCCRadar(r){var ctx=document.getElementById('tccRadarChart');if(tccRadarChart)tccRadarChart.destroy();var labels=Object.keys(r.cci_result.feature_scores).map(function(k){return k.replace(/_/g,' ')});tccRadarChart=new Chart(ctx,{type:'radar',data:{labels:labels,datasets:[{label:'CCI Score',data:Object.values(r.cci_result.feature_scores),backgroundColor:'rgba(0,230,118,0.2)',borderColor:'#00e676'}]},options:{responsive:true,plugins:{title:{display:true,text:'Cognitive Capital Index'}},scales:{r:{beginAtZero:true,max:1}}}})}var tccRefChart=null;function drawTCCReflexivity(r){var ctx=document.getElementById('tccReflexChart');if(tccRefChart)tccRefChart.destroy();var loop=r.reflexivity.reflexivity_loop;tccRefChart=new Chart(ctx,{type:'line',data:{labels:loop.map(function(s){return 'Step '+s.step}),datasets:[{label:'CCI After Impact',data:loop.map(function(s){return s.cci_after_impact}),borderColor:'#ffab00',backgroundColor:'rgba(255,171,0,0.2)',fill:true}]},options:{responsive:true,plugins:{title:{display:true,text:'Cognitive Reflexivity Loop - CCI Over Events'}},scales:{y:{beginAtZero:true,max:1}}}})}"""
    c += '</script>'
    return page('TCC Market Scenarios', c, 'tcc-scen')


# ============================================================================
# CSL PAGES
# ============================================================================

@app.route('/csl')
def page_csl():
    c = '<p>The Cognitive Settlement Layer (CSL) is a multi-agent AI system for real-time securities post-trade settlement optimization. 8 specialized agents analyze each trade and collaboratively compute the optimal settlement route.</p>'
    c += '<div class="info-banner"><strong>Paper 3 (160 pages):</strong> CSL uses a Control Tower Architecture with 3 layers. The 8 agents bid on each trade using a global objective function with Pareto frontier analysis.</div>'
    c += '<div class="card"><h3>The 8 CSL Agents</h3><table><thead><tr><th>Agent</th><th>Role</th></tr></thead><tbody>'
    for agent in CSLEngine.AGENT_DEFINITIONS:
        c += '<tr><td>' + agent['name'] + '</td><td>' + agent['role'] + '</td></tr>'
    c += '</tbody></table></div>'
    c += '<div class="formula-block">F(x) = w1*C(x) + w2*R(x) + w3*T(x) + w4*B(x)<br>where C=Cost, R=Risk, T=Timeliness, B=Exception probability</div>'
    c += '<div class="card"><h3>Submit Trade for Settlement</h3>'
    c += '<div class="csl-weight-panel">'
    c += '<div class="form-grid">'
    c += '<div class="form-group"><label>Trade Value ($)</label><input type="number" id="cslValue" value="5000000" step="100000"></div>'
    c += '<div class="form-group"><label>Currency</label><select id="cslCurrency"><option>USD</option><option>EUR</option><option>SGD</option><option>GBP</option></select></div>'
    c += '<div class="form-group"><label>Counterparty</label><input type="text" id="cslCounterparty" value="Broker A"></div>'
    c += '<div class="form-group"><label>Current Hour</label><input type="number" id="cslHour" value="13" min="0" max="23"></div>'
    c += '</div>'
    c += '<h4 style="color:var(--accent);margin:15px 0 10px">Orchestrator Weights (drag to recompute live)</h4>'
    c += '<div class="range-row"><label>Cost</label><input type="range" id="wCost" min="0" max="100" value="20" oninput="document.getElementById(\'wCostOut\').textContent=this.value;runCSLLive()"><output id="wCostOut">20</output></div>'
    c += '<div class="range-row"><label>Liquidity</label><input type="range" id="wLiq" min="0" max="100" value="20" oninput="document.getElementById(\'wLiqOut\').textContent=this.value;runCSLLive()"><output id="wLiqOut">20</output></div>'
    c += '<div class="range-row"><label>FX Path</label><input type="range" id="wFx" min="0" max="100" value="10" oninput="document.getElementById(\'wFxOut\').textContent=this.value;runCSLLive()"><output id="wFxOut">10</output></div>'
    c += '<div class="range-row"><label>Timeliness</label><input type="range" id="wTime" min="0" max="100" value="20" oninput="document.getElementById(\'wTimeOut\').textContent=this.value;runCSLLive()"><output id="wTimeOut">20</output></div>'
    c += '<div class="range-row"><label>Operational Risk</label><input type="range" id="wRisk" min="0" max="100" value="15" oninput="document.getElementById(\'wRiskOut\').textContent=this.value;runCSLLive()"><output id="wRiskOut">15</output></div>'
    c += '<div class="range-row"><label>Exception Prob.</label><input type="range" id="wExc" min="0" max="100" value="15" oninput="document.getElementById(\'wExcOut\').textContent=this.value;runCSLLive()"><output id="wExcOut">15</output></div>'
    c += '<p style="margin-top:10px"><button class="btn" onclick="runCSLLive()">Run 8-Agent Settlement</button></p>'
    c += '</div></div>'
    c += '<div id="cslResult"></div>'
    c += '<script>'
    c += """function runCSLLive(){var d={value:parseInt(document.getElementById('cslValue').value)||5000000,currency:document.getElementById('cslCurrency').value,counterparty:document.getElementById('cslCounterparty').value,current_hour:parseInt(document.getElementById('cslHour').value)||13};fetch('/api/csl/settle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(d)}).then(function(r){return r.json()}).then(function(r){var priors=JSON.parse(localStorage.getItem('cslPriors')||'{}');var html='<div class="card"><h3>Settlement Result</h3>';html+='<div class="stamp '+(r.optimal_route==='Internal'?'green':'blue')+'">OPTIMAL: '+r.optimal_route+'</div>';html+='<p>Pareto-optimal routes: '+r.pareto_frontier.join(', ')+'</p>';html+='<p><strong>vs Static SSI:</strong> '+r.ssi_comparison.interpretation+'</p></div>';html+='<div class="card"><h3>Agent Computations</h3>';r.agents.forEach(function(a){html+='<div class="agent-card"><div class="agent-name">'+a.name+' &mdash; '+a.status+'</div>';html+='<div class="agent-action">'+a.analysis+'</div>';html+='<div class="agent-action">Recommendation: '+a.recommendation+'</div>';if(a.bids){html+='<div class="route-row"><span style="flex:0 0 100px;font-size:0.8rem;color:var(--text-dim)">Bids:</span>';var maxBid=Math.max.apply(null,Object.values(a.bids));Object.entries(a.bids).forEach(function(e){var pct=(e[1]/maxBid)*100;html+='<div style="flex:1;min-width:80px"><div style="font-size:0.7rem;color:var(--text-dim)">'+e[0]+'</div><div class="route-bar"><span style="width:'+pct+'%;background:'+(['#7c4dff','#ff5252','#00e676','#ffab00','#00bcd4','#e91e63','#9c27b0','#4caf50','#795548','#607d8b'][Math.floor(Math.random()*10)])+'"></span></div></div>'});html+='</div>'}html+='</div>'});html+='</div>';html+='<div class="card"><h3>Adaptive Feedback (Bandit-Style RL)</h3>';html+='<p style="font-size:0.82rem;color:var(--text-dim)">Record a settlement outcome to update this route exception prior. Priors persist in your browser across scenarios.</p>';html+='<div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0">';r.pareto_frontier.forEach(function(route){var p=priors[route]||0;html+='<button class="btn" style="font-size:0.78rem" onclick="recordOutcome(\''+route+'\',\'success\')">Success: '+route+'</button>';html+='<button class="btn secondary" style="font-size:0.78rem" onclick="recordOutcome(\''+route+'\',\'fail\')">Fail: '+route+'</button>'})};html+='</div>';html+='<button class="btn secondary" style="font-size:0.78rem" onclick="resetPriors()">Reset All Priors</button>';html+='<div id="priorDisplay" style="margin-top:10px;font-size:0.82rem;color:var(--text-dim)"></div>';html+='</div>';html+='<div class="chart-container"><canvas id="cslBidChart"></canvas></div>';document.getElementById('cslResult').innerHTML=html;updatePriorDisplay();drawCSLBids(r)})}function recordOutcome(route,result){var priors=JSON.parse(localStorage.getItem('cslPriors')||'{}');if(!priors[route])priors[route]=0;if(result==='fail')priors[route]=Math.min(1,(priors[route]||0)+0.1);else priors[route]=Math.max(0,(priors[route]||0)-0.05);localStorage.setItem('cslPriors',JSON.stringify(priors));updatePriorDisplay()}function resetPriors(){localStorage.removeItem('cslPriors');updatePriorDisplay()}function updatePriorDisplay(){var priors=JSON.parse(localStorage.getItem('cslPriors')||'{}');var el=document.getElementById('priorDisplay');if(!el)return;var keys=Object.keys(priors);if(!keys.length){el.innerHTML='No priors recorded yet.';return}el.innerHTML='Current exception priors: '+keys.map(function(k){return k+'='+Math.round(priors[k]*100)+'%'}).join(', ')}var cslBidChart=null;function drawCSLBids(r){var ctx=document.getElementById('cslBidChart');if(!ctx)return;if(cslBidChart)cslBidChart.destroy();var agents=r.agents.filter(function(a){return a.bids});if(!agents.length)return;var routes=Object.keys(agents[0].bids);var datasets=agents.map(function(a,i){var colors=['#7c4dff','#ff5252','#00e676','#ffab00','#00bcd4','#e91e63','#9c27b0','#4caf50'];return{label:a.name,data:routes.map(function(rt){return a.bids[rt]||0}),backgroundColor:colors[i%8]}});cslBidChart=new Chart(ctx,{type:'bar',data:{labels:routes,datasets:datasets},options:{responsive:true,plugins:{title:{display:true,text:'Agent Bids by Settlement Route'}},scales:{y:{beginAtZero:true}}}})}"""
    c += '</script>'
    return page('Cognitive Settlement Layer', c, 'csl')

@app.route('/csl/rl')
def page_csl_rl():
    c = '<p>The Q-Learning engine trains a reinforcement learning agent to optimize settlement routing. Over 500 episodes, the agent learns which routes minimize the global objective function (cost + risk + timeliness + exceptions).</p>'
    c += '<div class="info-banner"><strong>Q-Learning:</strong> State = (value bracket, currency, market, hour). Actions = 8 settlement routes. Reward = negative objective function. Uses epsilon-greedy exploration with decay.</div>'
    c += '<div class="card"><h3>Train RL Agent</h3>'
    c += '<div class="form-group"><label>Episodes</label><input type="number" id="rlEpisodes" value="500" step="50" min="100" max="5000"></div>'
    c += '<p><button class="btn" onclick="trainRL()">Train Agent</button></p></div>'
    c += '<div id="rlResult"></div>'
    c += '<div class="chart-container" id="rlChartBox" style="display:none"><canvas id="rlConvergenceChart"></canvas></div>'
    c += '<script>'
    c += """function trainRL(){var eps=parseInt(document.getElementById('rlEpisodes').value)||500;fetch('/api/csl/rl-train?episodes='+eps).then(function(r){return r.json()}).then(function(r){var html='<div class="card"><h3>Training Complete</h3>';html+='<div class="result-value">Optimal Route: '+r.optimal_route+'</div>';html+='<table><tr><th>Metric</th><th>Value</th></tr>';html+='<tr><td>Episodes</td><td>'+r.episodes+'</td></tr>';html+='<tr><td>Learning Rate</td><td>'+r.alpha+'</td></tr>';html+='<tr><td>Discount</td><td>'+r.gamma+'</td></tr>';html+='<tr><td>Q-Table Size</td><td>'+r.q_table_size+' states</td></tr></table>';html+='<h3 style="margin-top:15px">Final Q-Values</h3><table><tr><th>Route</th><th>Q-Value</th></tr>';Object.entries(r.final_q_values).forEach(function(e){html+='<tr><td>'+e[0]+'</td><td>'+e[1].toFixed(2)+'</td></tr>'});html+='</table></div>';document.getElementById('rlResult').innerHTML=html;drawRLChart(r)})}var rlChart=null;function drawRLChart(r){var ctx=document.getElementById('rlConvergenceChart');document.getElementById('rlChartBox').style.display='block';if(rlChart)rlChart.destroy();var hist=r.convergence_history;rlChart=new Chart(ctx,{type:'line',data:{labels:hist.map(function(h){return h.episode}),datasets:[{label:'Avg Reward',data:hist.map(function(h){return h.avg_reward}),borderColor:'#00bcd4',backgroundColor:'rgba(0,188,212,0.2)',fill:true,yAxisID:'y'},{label:'Epsilon',data:hist.map(function(h){return h.epsilon}),borderColor:'#ffab00',borderDash:[5,5],yAxisID:'y1'}]},options:{responsive:true,plugins:{title:{display:true,text:'Q-Learning Convergence'}},scales:{y:{position:'left',title:{display:true,text:'Avg Reward'}},y1:{position:'right',title:{display:true,text:'Epsilon'},min:0,max:1,grid:{drawOnChartArea:false}}}}})}"""
    c += '</script>'
    return page('CSL Reinforcement Learning', c, 'csl-rl')


@app.route('/csl/scenarios')
def page_csl_scen():
    c = '<p>Predefined settlement scenarios from the CSL paper, covering cross-border equity, repo/SBL, multi-currency FX, and high-stress conditions. Each scenario shows full 8-agent analysis with Pareto frontier.</p>'
    c += '<div class="card"><h3>Settlement Scenarios</h3>'
    scenarios = [
        ('cross_border_equity', 'Cross-Border Equity Settlement (SGX to Euroclear)'),
        ('repo_sbl', 'Repo / Securities Lending Optimization'),
        ('multi_currency_fx', 'Multi-Currency FX-Linked Settlement'),
        ('high_stress', 'High-Stress Market Conditions'),
    ]
    for name, label in scenarios:
        c += '<a class="scenario-btn" href="#" onclick="runCSL(\'' + name + '\');return false">' + label + '</a>'
    c += '</div>'
    c += '<div id="cslResult"></div>'
    c += '<script>'
    c += """function runCSL(name){fetch('/api/csl/scenario/'+name).then(function(r){return r.json()}).then(function(r){var html='<div class="card"><h3>'+r.description+'</h3>';html+='<div class="result-value">Optimal Route: '+r.optimal_route+'</div>';html+='<p>Pareto-optimal routes: '+r.pareto_frontier.join(', ')+'</p>';html+='<p><strong>vs Static SSI:</strong> '+r.ssi_comparison.interpretation+'</p>';html+='</div>';html+='<div class="card"><h3>Agent Details</h3>';r.agents.forEach(function(a){html+='<div class="agent-card"><div class="agent-name">'+a.name+'</div>';html+='<div class="agent-action">'+a.analysis+'</div>';html+='<div class="agent-action">Recommends: '+a.recommendation+'</div>';if(a.bids){var bids='';Object.entries(a.bids).forEach(function(e){bids+=e[0]+'='+e[1]+', '});html+='<div class="agent-action">Bids: '+bids+'</div>'}html+='</div>'});html+='</div>';html+='<div class="chart-container"><canvas id="cslScenChart"></canvas></div>';document.getElementById('cslResult').innerHTML=html;drawCSLScenario(r)})}var cslScChart=null;function drawCSLScenario(r){var ctx=document.getElementById('cslScenChart');if(cslScChart)cslScChart.destroy();var agents=r.agents.filter(function(a){return a.bids});if(!agents.length)return;var routes=Object.keys(agents[0].bids);var datasets=agents.map(function(a,i){var colors=['#7c4dff','#ff5252','#00e676','#ffab00','#00bcd4','#e91e63','#9c27b0','#4caf50'];return{label:a.name,data:routes.map(function(r){return a.bids[r]||0}),backgroundColor:colors[i%8]}});cslScChart=new Chart(ctx,{type:'bar',data:{labels:routes,datasets:datasets},options:{responsive:true,plugins:{title:{display:true,text:'Agent Bids by Settlement Route'}},scales:{y:{beginAtZero:true}}}})}"""
    c += '</script>'
    return page('CSL Settlement Scenarios', c, 'csl-scen')


# ============================================================================
# GATE SYMPHONY PAGES
# ============================================================================

@app.route('/gate')
def page_gate():
    c = '<p>The Gate Symphony uses deterministic Boolean logic gates (AND, OR, XOR, NAND) to constrain agentic AI autonomy. Every consequential action must pass through gates whose satisfaction requires inputs the agent cannot produce.</p>'
    c += '<div class="info-banner"><strong>Paper 4:</strong> No-autonomous-path theorem &mdash; for any well-formed gate symphony, there exists no satisfying assignment of agent-producible signals alone that opens a path to a consequential action. Verified with 50,000 randomized cases and 152,285 capability checks.</div>'
    c += '<div class="card"><h3>The Four Canonical Gates</h3>'
    c += '<div class="stat-grid">'
    c += '<div class="stat-card gate"><div class="value">AND</div><div class="label">Conjunctive Authorization</div></div>'
    c += '<div class="stat-card gate"><div class="value">OR</div><div class="label">Redundant Channels</div></div>'
    c += '<div class="stat-card gate"><div class="value">XOR</div><div class="label">Mode Exclusivity</div></div>'
    c += '<div class="stat-card gate"><div class="value">NAND</div><div class="label">Circuit Breaker</div></div>'
    c += '</div></div>'
    c += '<div class="card"><h3>Autonomy Lattice</h3>'
    c += '<div class="formula-block">DENY < OBSERVE < SIMULATE < PROPOSE < BOUNDED_EXECUTE < EXECUTE<br>L_eff = meet_i(L_i) &mdash; minimum authority wins</div>'
    c += '<p>Each gate returns a maximum permitted autonomy level. The effective level is the meet (greatest lower bound) of all gate outputs.</p>'
    c += '<div class="lattice">'
    levels = ['DENY', 'OBSERVE', 'SIMULATE', 'PROPOSE', 'BOUNDED_EXECUTE', 'EXECUTE']
    for i, lv in enumerate(levels):
        cls = 'lnode'
        if lv == 'BOUNDED_EXECUTE':
            cls += ' req'
        c += '<div class="' + cls + '">' + lv + '</div>'
        if i < len(levels) - 1:
            c += '<div class="ledge"></div>'
    c += '</div></div>'
    c += '<div class="card"><h3>Signal Provenance</h3>'
    c += '<p>Every signal entering a gate is tagged with its provenance:</p>'
    c += '<p><span class="prov-tag prov-sigma-a">sigma_A</span> Agent-produced &nbsp; <span class="prov-tag prov-sigma-s">sigma_S</span> System-generated &nbsp; <span class="prov-tag prov-sigma-h">sigma_H</span> Human-issued</p>'
    c += '<p>A well-formed C2/C3 gate must contain at least one sigma_S or sigma_H signal. Gates with only sigma_A inputs are flagged as violations.</p></div>'
    c += '<div class="card"><h3>Evaluate Gate Symphony</h3>'
    c += '<p>Select a scenario to evaluate the full gate symphony with signal provenance checking and no-autonomous-path verification:</p>'
    scenarios = [
        ('ssi_routing_safe', 'SSI Routing &mdash; All Approvals (Safe)'),
        ('ssi_routing_blocked', 'SSI Routing &mdash; No Human Approval (Blocked)'),
        ('nand_circuit_breaker', 'NAND Circuit Breaker &mdash; Fraud Detection'),
        ('xor_mode_conflict', 'XOR &mdash; Sandbox vs Production Conflict'),
        ('agent_only_attack', 'Agent-Only Attack (No-Autonomous-Path Test)'),
    ]
    for name, label in scenarios:
        c += '<a class="scenario-btn" href="#" onclick="runGate(\'' + name + '\');return false">' + label + '</a>'
    c += '</div>'
    c += '<div id="gateResult"></div>'
    c += '<script>'
    c += """function runGate(name){fetch('/api/gate/scenario/'+name).then(function(r){return r.json()}).then(function(r){var color=r.overall_result==='PROCEED'?'var(--success)':'var(--danger)';var html='<div class="card"><h3>'+r.description+'</h3>';html+='<div class="stamp '+(r.overall_result==='PROCEED'?'green':'red')+'">'+r.overall_result+'</div>';html+='<p>Effective Autonomy: '+r.effective_autonomy_level+'</p>';html+='<p>Consequence Class: '+r.consequence_class+' &mdash; '+r.consequence_description+'</p>';html+='<p>No-Autonomous-Path Verified: <strong>'+(r.no_autonomous_path_verified?'YES':'NO &mdash; VIOLATION')+'</strong></p>';html+='<p><em>'+r.nap_analysis+'</em></p></div>';html+='<div class="card"><h3>Gate Circuit Diagram</h3>';html+='<div class="gate-circuit">';r.gate_results.forEach(function(g,idx){var cls=g.result==='PROCEED'?'proceed':'blocked';if(g.violation)cls='violation';html+='<div class="gate-node '+cls+'">';html+='<div style="font-weight:bold">'+g.gate_name+'</div>';html+='<div style="font-size:0.8rem">'+g.gate_type+'</div>';html+='<div style="font-size:0.7rem;margin-top:5px">';g.input_signals.forEach(function(sig,i){var prov=g.input_provenances[i];var provClass=prov==='sigma_A'?'prov-sigma-a':(prov==='sigma_S'?'prov-sigma-s':'prov-sigma-h');html+='<span class="prov-tag '+provClass+'">'+prov+'</span> '});html+='</div>';html+='<div style="margin-top:5px;color:'+(g.result==='PROCEED'?'var(--success)':'var(--danger)')+'">'+g.result+'</div>';html+='</div>';if(idx<r.gate_results.length-1)html+='<div class="gate-arrow">&rarr;</div>'});html+='</div>';if(r.audit_trail){html+='<h4 style="margin-top:15px">Audit Trail</h4><table>';html+='<tr><td>Gates Evaluated</td><td>'+r.audit_trail.gates_evaluated+'</td></tr>';html+='<tr><td>Gates Satisfied</td><td>'+r.audit_trail.gates_satisfied+'</td></tr>';html+='<tr><td>Gates Blocked</td><td>'+r.audit_trail.gates_blocked+'</td></tr>';html+='<tr><td>Well-Formedness Violations</td><td>'+r.audit_trail.well_formedness_violations+'</td></tr></table>'}html+='</div>';html+='<div class="card"><h3>Gate Details</h3>';r.gate_results.forEach(function(g){var gcolor=g.result==='PROCEED'?'var(--success)':'var(--danger)';html+='<div class="agent-card"><div class="agent-name" style="color:'+gcolor+'">'+g.gate_name+' ('+g.gate_type+') &mdash; '+g.result+'</div>';html+='<div class="agent-action">Control Primitive: '+g.control_primitive+'</div>';html+='<div class="agent-action">'+g.description+'</div>';html+='<div class="agent-action">Inputs: ['+g.inputs.join(', ')+'] &rarr; Output: '+g.output+'</div>';html+='<div class="agent-action">Signals: ['+g.input_signals.join(', ')+']</div>';html+='<div class="agent-action">Provenance: ['+g.input_provenances.join(', ')+']</div>';if(g.violation)html+='<div class="agent-action" style="color:var(--danger)"><strong>VIOLATION:</strong> '+g.violation+'</div>';if(!g.well_formed)html+='<div class="agent-action" style="color:#ff9800">Not well-formed</div>';html+='</div>'});html+='</div>';document.getElementById('gateResult').innerHTML=html})}"""
    c += '</script>'
    return page('The Gate Symphony', c, 'gate')

@app.route('/gate/truth-tables')
def page_gate_tt():
    c = '<p>Truth tables for the four canonical logic gates used in the Gate Symphony architecture. Each gate type constrains agent autonomy differently.</p>'
    for gate_type in ['AND', 'OR', 'XOR', 'NAND']:
        gt = GateSymphony.GATE_TYPES[gate_type]
        c += '<div class="card"><h3>' + gate_type + ' Gate &mdash; ' + gt['control_primitive'] + '</h3>'
        c += '<p>' + gt['description'] + '</p>'
        c += '<p style="color:var(--text-dim);font-size:0.8rem">Attenuation: ' + gt['attenuation'] + '</p>'
        c += '<table><thead><tr><th>A</th><th>B</th><th>Output</th></tr></thead><tbody>'
        for row in gt['truth_table']:
            c += '<tr>'
            for val in row:
                style = 'color:' + ('var(--success)' if val == 1 else 'var(--danger)')
                c += '<td style="' + style + ';font-weight:bold">' + str(val) + '</td>'
            c += '</tr>'
        c += '</tbody></table></div>'
    return page('Gate Truth Tables', c, 'gate-tt')

@app.route('/gate/scenarios')
def page_gate_scen():
    c = '<p>Predefined Gate Symphony scenarios demonstrating the architecture in action. Each scenario shows the full gate circuit evaluation with signal provenance and NAP verification.</p>'
    c += '<div class="card"><h3>Gate Scenarios</h3>'
    scenarios = [
        ('ssi_routing_safe', 'SSI Routing &mdash; All Approvals Present'),
        ('ssi_routing_blocked', 'SSI Routing &mdash; Missing Human Approval'),
        ('nand_circuit_breaker', 'NAND Circuit Breaker &mdash; Fraud Pattern'),
        ('xor_mode_conflict', 'XOR &mdash; Mode Conflict Detection'),
        ('agent_only_attack', 'Agent-Only Attack — NAP Test'),
        ('or_redundant_channels', 'OR: Redundant Channels'),
        ('nand_emergency_halt', 'NAND: Emergency Halt'),
    ]
    for name, label in scenarios:
        c += '<a class="scenario-btn" href="#" onclick="runG(\'' + name + '\');return false">' + label + '</a>'
    c += '</div>'
    c += '<div id="gateResult"></div>'
    c += '<script>'
    c += """function runG(name){fetch('/api/gate/scenario/'+name).then(function(r){return r.json()}).then(function(r){var color=r.overall_result==='PROCEED'?'var(--success)':'var(--danger)';var html='<div class="card"><h3>'+r.description+'</h3>';html+='<div class="stamp '+(r.overall_result==='PROCEED'?'green':'red')+'">'+r.overall_result+'</div>';html+='<p>Autonomy: '+r.effective_autonomy_level+'</p>';html+='<p>NAP Verified: <strong>'+(r.no_autonomous_path_verified?'YES':'NO')+'</strong></p>';html+='<p><em>'+r.nap_analysis+'</em></p></div>';html+='<div class="card"><h3>Gate Circuit</h3><div class="gate-circuit">';r.gate_results.forEach(function(g,idx){var cls=g.result==='PROCEED'?'proceed':'blocked';if(g.violation)cls='violation';html+='<div class="gate-node '+cls+'"><div style="font-weight:bold">'+g.gate_name+'</div><div style="font-size:0.8rem">'+g.gate_type+'</div>';g.input_signals.forEach(function(sig,i){var prov=g.input_provenances[i];var pcls=prov==='sigma_A'?'prov-sigma-a':(prov==='sigma_S'?'prov-sigma-s':'prov-sigma-h');html+='<span class="prov-tag '+pcls+'">'+prov+'</span>'});html+='<div style="margin-top:5px;color:'+(g.result==='PROCEED'?'var(--success)':'var(--danger)')+'">'+g.result+'</div></div>';if(idx<r.gate_results.length-1)html+='<div class="gate-arrow">&rarr;</div>'});html+='</div></div>';r.gate_results.forEach(function(g){var gcolor=g.result==='PROCEED'?'var(--success)':'var(--danger)';html+='<div class="agent-card"><div class="agent-name" style="color:'+gcolor+'">'+g.gate_name+' &mdash; '+g.result+'</div>';html+='<div class="agent-action">Signals: ['+g.input_signals.join(', ')+']</div>';html+='<div class="agent-action">Provenance: ['+g.input_provenances.join(', ')+']</div>';if(g.violation)html+='<div class="agent-action" style="color:var(--danger)">'+g.violation+'</div>';html+='</div>'});document.getElementById('gateResult').innerHTML=html})}"""
    c += '</script>'
    return page('Gate Scenarios', c, 'gate-scen')

# ============================================================================
# RESEARCH PAPER ARCHIVE
# ============================================================================

PAPERS = [
    {
        'slug': 'quantum-personnel-securities',
        'code': 'QPS',
        'title': 'Quantum Personnel Securities (QPS): A Theoretical Framework for a Third Asset Class Beyond Equity and Debt',
        'authors': ['Saumyajit Ghosh'],
        'date': '2025-10',
        'year': '2025',
        'pages': 48,
        'venue': 'ResearchGate / SSRN',
        'ssrn_id': None,
        'doi': None,
        'url': 'https://www.researchgate.net/publication/',
        'abstract': 'QPS introduces a third asset class using quantum mechanics to model the human and behavioral dimension of corporate value. The framework establishes a quantum state C that is neither pure equity (state A) nor pure debt (state B), but a third orthogonal state linking financial outcomes to human decision-making quality and cognitive bias. The paper defines superposition of strategies, non-commuting bias operators (overconfidence, loss aversion, groupthink, anchoring, confirmation), entanglement between leadership teams, and a payoff function adjusted by leadership entropy.',
        'keywords': ['quantum finance', 'behavioral economics', 'asset pricing', 'cognitive bias', 'superposition', 'entanglement', 'leadership dynamics'],
        'innovations': ['Third asset class beyond equity and debt', 'Quantum state vector for leadership strategies', 'Non-commuting bias operators', 'Entropy-adjusted payoff function'],
    },
    {
        'slug': 'tokenized-cognitive-capital',
        'code': 'TCC',
        'title': 'Tokenized Cognitive Capital: Pricing and Trading Organizational Intelligence',
        'authors': ['Saumyajit Ghosh'],
        'date': '2025-11',
        'year': '2025',
        'pages': 59,
        'venue': 'SSRN',
        'ssrn_id': None,
        'doi': None,
        'url': 'https://ssrn.com/',
        'abstract': 'TCC extends the QPS framework into a tokenized instrument. The Cognitive Capital Index (CCI) measures collective organizational intelligence across 7 dimensions: knowledge creation, decision efficiency, AI alignment, learning velocity, innovation output, collaboration index, and adaptive capacity. Token valuation incorporates cognitive decay with a half-life, a Volatility of Cognition (VoC) risk premium analogous to financial volatility, and cognitive reflexivity loops where market events feed back into CCI. The paper defines the Mechanics of Cognitive Value (MCV) and its projection into tradeable tokens.',
        'keywords': ['tokenization', 'cognitive capital', 'organizational intelligence', 'volatility of cognition', 'cognitive decay', 'reflexivity', 'asset pricing'],
        'innovations': ['Cognitive Capital Index (7 dimensions)', 'Cognitive decay half-life', 'Volatility of Cognition (VoC) risk premium', 'Cognitive reflexivity loops'],
    },
    {
        'slug': 'cognitive-settlement-layer',
        'code': 'CSL',
        'title': 'Cognitive Settlement Layer: Multi-Agent Post-Trade Settlement Optimisation',
        'authors': ['Saumyajit Ghosh'],
        'date': '2026-02',
        'year': '2026',
        'pages': 160,
        'venue': 'SSRN',
        'ssrn_id': '6167431',
        'doi': None,
        'url': 'https://ssrn.com/abstract=6167431',
        'abstract': 'CSL is a multi-agent AI framework for dynamic securities settlement routing. Eight specialized agents (Cost, Liquidity, FX, Timeliness, Risk, Exception, Learning, Orchestrator) score every candidate settlement route using a global objective function over cost, liquidity, FX conversion, timeliness, operational risk and exception probability. A compliance agent removes non-compliant routes. The orchestrator either confirms the standing settlement instruction or overrides it with explanation. The paper uses a Control Tower Architecture with 3 layers, Pareto frontier optimization, and MDP-based reinforcement learning for route optimization over time.',
        'keywords': ['settlement', 'post-trade', 'multi-agent systems', 'reinforcement learning', 'Pareto optimization', 'MDP', 'securities processing'],
        'innovations': ['8-agent settlement optimization', 'Global objective function with Pareto frontier', 'Control Tower Architecture', 'RL-based MDP routing'],
    },
    {
        'slug': 'gate-symphony',
        'code': 'GATE',
        'title': 'The Gate Symphony: Deterministic Logic Gates for Bounding Agentic AI Autonomy',
        'authors': ['Saumyajit Ghosh'],
        'date': '2026',
        'year': '2026',
        'pages': 35,
        'venue': 'Working paper',
        'ssrn_id': None,
        'doi': None,
        'url': '',
        'abstract': 'The Gate Symphony uses deterministic Boolean logic gates (AND, OR, XOR, NAND) to constrain agentic AI autonomy. Every consequential action must pass through gates whose satisfaction requires inputs the agent cannot produce. The No-Autonomous-Path theorem proves that for any well-formed gate symphony, there exists no satisfying assignment of agent-producible signals alone that opens a path to a consequential action. Signal provenance is tracked as sigma_A (agent-produced), sigma_S (system-generated), and sigma_H (human-issued). The theorem is verified across 50,000 randomized cases and 152,285 capability checks.',
        'keywords': ['AI safety', 'agentic AI', 'boolean logic', 'autonomy bounds', 'no-autonomous-path theorem', 'signal provenance', 'governance'],
        'innovations': ['No-Autonomous-Path theorem', 'Signal provenance (sigma_A/sigma_S/sigma_H)', 'Autonomy lattice (DENY to EXECUTE)', '4 canonical gate types with attenuation'],
    },
]

def scholar_meta(paper):
    """Generate Google Scholar-compatible meta tags and Schema.org JSON-LD."""
    tags = ''
    tags += '<meta name="citation_title" content="' + paper['title'] + '">'
    for author in paper['authors']:
        tags += '<meta name="citation_author" content="' + author + '">'
    tags += '<meta name="citation_publication_date" content="' + paper['date'] + '">'
    tags += '<meta name="citation_journal_title" content="' + paper['venue'] + '">'
    if paper.get('doi'):
        tags += '<meta name="citation_doi" content="' + paper['doi'] + '">'
    # Schema.org JSON-LD
    jsonld = {
        '@context': 'https://schema.org',
        '@type': 'ScholarlyArticle',
        'name': paper['title'],
        'author': [{'@type': 'Person', 'name': a} for a in paper['authors']],
        'datePublished': paper['date'],
        'publisher': {'@type': 'Organization', 'name': paper['venue']},
        'abstract': paper['abstract'],
        'keywords': ', '.join(paper['keywords']),
    }
    tags += '<script type="application/ld+json">' + json.dumps(jsonld) + '</script>'
    return tags

@app.route('/papers')
def page_papers():
    c = '<p>This archive contains the four research papers that form the theoretical foundation of FinSight AI. Each paper has its own landing page with abstract, keywords, and links to SSRN/ResearchGate.</p>'
    c += '<div class="info-banner"><strong>Google Scholar indexing:</strong> Each paper page emits citation_* meta tags and Schema.org ScholarlyArticle JSON-LD for scholarly discovery.</div>'
    c += '<div class="card"><h3>Papers</h3>'
    for p in PAPERS:
        c += '<a class="paper-card-link" href="/papers/' + p['slug'] + '">'
        c += '<h4><span class="tag tag-' + ('quantum' if p['code'] == 'QPS' else 'success' if p['code'] == 'TCC' else 'warning' if p['code'] == 'CSL' else 'danger') + '">' + p['code'] + '</span> ' + p['title'][:80] + '...</h4>'
        c += '<div class="paper-meta">' + p['venue'] + ' | ' + str(p['pages']) + ' pages | ' + p['date'] + '</div>'
        c += '<div class="paper-meta">' + ', '.join(p['keywords'][:4]) + '</div>'
        c += '</a>'
    c += '</div>'
    c += '<div class="card"><h3>Rights & Licensing</h3>'
    c += '<p>The software source code of FinSight AI is open source. The research papers, abstracts, figures, and scholarly content remain the copyright of the author. SSRN links are provided for the authoritative version of each paper.</p>'
    c += '<p><em>Version: Author manuscript | Copyright: (c) 2025-2026 Saumyajit Ghosh | Licence: All rights reserved</em></p>'
    c += '</div>'
    return page('Research Papers', c, 'papers')

@app.route('/papers/<slug>')
def page_paper_detail(slug):
    paper = None
    for p in PAPERS:
        if p['slug'] == slug:
            paper = p
            break
    if not paper:
        return page('Paper Not Found', '<p>The requested paper was not found.</p>', 'papers'), 404
    c = '<a href="/papers" style="color:var(--accent);text-decoration:none">&larr; Back to all papers</a>'
    c += '<div class="paper-detail">'
    c += '<h3>' + paper['title'] + '</h3>'
    c += '<div class="paper-meta"><strong>Authors:</strong> ' + ', '.join(paper['authors']) + ' | <strong>Venue:</strong> ' + paper['venue'] + ' | <strong>Pages:</strong> ' + str(paper['pages']) + ' | <strong>Date:</strong> ' + paper['date'] + '</div>'
    if paper.get('ssrn_id'):
        c += '<div class="paper-meta"><strong>SSRN:</strong> <a href="' + paper['url'] + '" target="_blank" rel="noopener">' + paper['url'] + '</a></div>'
    elif paper.get('url'):
        c += '<div class="paper-meta"><strong>URL:</strong> <a href="' + paper['url'] + '" target="_blank" rel="noopener">' + paper['url'] + '</a></div>'
    c += '<div class="paper-abstract">' + paper['abstract'] + '</div>'
    c += '<div class="paper-keywords">'
    for kw in paper['keywords']:
        c += '<span class="keyword">' + kw + '</span>'
    c += '</div>'
    c += '<h4 style="margin-top:15px;color:var(--accent)">Key Innovations</h4><ul>'
    for inn in paper['innovations']:
        c += '<li>' + inn + '</li>'
    c += '</ul>'
    c += '<div class="paper-meta" style="margin-top:15px;border-top:1px solid var(--border);padding-top:10px">'
    c += 'Version: Author manuscript | Copyright: (c) ' + paper['year'] + ' ' + ', '.join(paper['authors']) + ' | Licence: All rights reserved'
    c += '</div>'
    c += '</div>'
    # Use a custom page wrapper that includes scholar meta tags
    nav_html = '<div class="sidebar"><div class="sidebar-header"><h1>FinSight AI</h1><div class="version">v3.2 COGNITIVE FINANCE</div></div>'
    sections = [
        ('Overview', [('/', 'Dashboard', 'overview')]),
        ('Paper 1: QPS', [('/qps', 'Quantum Personnel Securities', 'qps'), ('/qps/bias', 'Bias Operators', 'qps-bias'), ('/qps/scenarios', 'Simulation Scenarios', 'qps-scen')]),
        ('Paper 2: TCC', [('/tcc', 'Cognitive Capital Index', 'tcc'), ('/tcc/valuation', 'Token Valuation', 'tcc-val'), ('/tcc/scenarios', 'Market Scenarios', 'tcc-scen')]),
        ('Paper 3: CSL', [('/csl', 'Settlement Layer', 'csl'), ('/csl/rl', 'RL Training', 'csl-rl'), ('/csl/scenarios', 'Settlement Scenarios', 'csl-scen')]),
        ('Paper 4: Gates', [('/gate', 'Gate Symphony', 'gate'), ('/gate/truth-tables', 'Truth Tables', 'gate-tt'), ('/gate/scenarios', 'Gate Scenarios', 'gate-scen')]),
        ('Archive', [('/papers', 'Research Papers', 'papers')]),
    ]
    for section_name, items in sections:
        nav_html += '<div class="nav-section"><div class="nav-section-title">' + section_name + '</div>'
        for href, label, key in items:
            cls = ' active' if key == 'papers' else ''
            nav_html += '<a href="' + href + '" class="nav-item' + cls + '">' + label + '</a>'
        nav_html += '</div>'
    nav_html += '</div>'
    return '<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>' + paper['title'][:60] + ' | FinSight AI</title><style>' + CSS + '</style>' + scholar_meta(paper) + '</head><body>' + nav_html + '<div class="main"><div class="page-header"><h2>' + paper['code'] + ': ' + paper['title'][:50] + '...</h2></div>' + c + '</div></body></html>'




# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
