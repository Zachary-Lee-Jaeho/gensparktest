"""
AST Alignment for VEGA-Verified.

Implements GumTree-style AST alignment for comparing multiple
reference implementations and extracting common patterns.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
import re
from collections import defaultdict


class AlignmentType(Enum):
    """Types of AST node alignment."""
    IDENTICAL = "identical"           # Exactly the same
    EQUIVALENT = "equivalent"         # Same structure, different values
    TARGET_SPECIFIC = "target_specific"  # Differs across targets
    INSERTED = "inserted"             # Present in one, not others
    DELETED = "deleted"               # Missing in one
    MODIFIED = "modified"             # Same position, different content


@dataclass
class ASTNode:
    """Simple AST node for alignment."""
    node_type: str
    label: str
    value: str = ""
    children: List['ASTNode'] = field(default_factory=list)
    parent: Optional['ASTNode'] = None
    source_target: str = ""
    line_number: int = 0
    
    # Alignment metadata
    aligned_with: List['ASTNode'] = field(default_factory=list)
    alignment_type: AlignmentType = AlignmentType.IDENTICAL
    
    @property
    def height(self) -> int:
        """Calculate tree height."""
        if not self.children:
            return 1
        return 1 + max(c.height for c in self.children)
    
    @property
    def size(self) -> int:
        """Calculate subtree size."""
        return 1 + sum(c.size for c in self.children)
    
    def add_child(self, child: 'ASTNode') -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)
    
    def descendants(self) -> List['ASTNode']:
        """Get all descendants."""
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.descendants())
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.node_type,
            "label": self.label,
            "value": self.value[:50] if self.value else "",
            "children": [c.to_dict() for c in self.children],
            "alignment": self.alignment_type.value,
        }
    
    def __hash__(self):
        return hash((self.node_type, self.label, self.value))
    
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (self.node_type == other.node_type and 
                self.label == other.label and
                self.value == other.value)


@dataclass
class AlignmentMapping:
    """Mapping between aligned AST nodes."""
    source: ASTNode
    target: ASTNode
    alignment_type: AlignmentType
    similarity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": f"{self.source.node_type}:{self.source.label}",
            "target": f"{self.target.node_type}:{self.target.label}",
            "type": self.alignment_type.value,
            "similarity": self.similarity,
        }


@dataclass
class MultiAlignmentResult:
    """Result of aligning multiple ASTs."""
    asts: List[ASTNode]
    source_targets: List[str]
    alignments: List[List[AlignmentMapping]]  # Pairwise alignments
    common_nodes: List[ASTNode]  # Nodes present in all
    target_specific_nodes: Dict[str, List[ASTNode]]  # Target -> specific nodes
    similarity_matrix: List[List[float]] = field(default_factory=list)
    
    @property
    def overall_similarity(self) -> float:
        """Calculate overall alignment similarity."""
        if not self.similarity_matrix:
            return 0.0
        
        total = sum(sum(row) for row in self.similarity_matrix)
        n = len(self.similarity_matrix)
        if n <= 1:
            return 1.0
        
        # Exclude diagonal
        return total / (n * (n - 1)) if n > 1 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "targets": self.source_targets,
            "overall_similarity": self.overall_similarity,
            "common_nodes": len(self.common_nodes),
            "target_specific": {k: len(v) for k, v in self.target_specific_nodes.items()},
        }


class ASTAligner:
    """
    GumTree-style AST alignment algorithm.
    
    Aligns ASTs from multiple implementations to:
    1. Identify common (target-independent) structures
    2. Identify target-specific variations
    3. Extract abstract patterns
    """
    
    # Similarity thresholds
    MIN_HEIGHT = 2  # Minimum height for top-down matching
    MIN_DICE = 0.5  # Minimum Dice coefficient for matching
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def parse_to_ast(self, code: str, target: str) -> ASTNode:
        """
        Parse code into AST representation.
        
        Args:
            code: Source code
            target: Target name (e.g., "RISCV", "ARM")
        
        Returns:
            Root AST node
        """
        root = ASTNode(
            node_type="function",
            label="root",
            source_target=target
        )
        
        lines = code.split('\n')
        self._parse_lines(lines, root, target)
        
        return root
    
    def _parse_lines(
        self,
        lines: List[str],
        parent: ASTNode,
        target: str
    ) -> None:
        """Parse lines into AST nodes."""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if not line or line.startswith('//'):
                i += 1
                continue
            
            # Switch statement
            if line.startswith('switch'):
                node, consumed = self._parse_switch_ast(lines, i, target)
                parent.add_child(node)
                i += consumed
            
            # If statement
            elif line.startswith('if'):
                node, consumed = self._parse_if_ast(lines, i, target)
                parent.add_child(node)
                i += consumed
            
            # For/while loop
            elif line.startswith(('for', 'while')):
                node, consumed = self._parse_loop_ast(lines, i, target)
                parent.add_child(node)
                i += consumed
            
            # Return statement
            elif line.startswith('return'):
                node = ASTNode(
                    node_type="return",
                    label="return",
                    value=self._extract_return_value(line),
                    source_target=target,
                    line_number=i
                )
                parent.add_child(node)
                i += 1
            
            # Assignment
            elif '=' in line and not line.startswith('if'):
                node = ASTNode(
                    node_type="assignment",
                    label="assign",
                    value=line,
                    source_target=target,
                    line_number=i
                )
                parent.add_child(node)
                i += 1
            
            # Function call
            elif '(' in line and line.endswith(';'):
                node = ASTNode(
                    node_type="call",
                    label=self._extract_func_name(line),
                    value=line,
                    source_target=target,
                    line_number=i
                )
                parent.add_child(node)
                i += 1
            
            else:
                i += 1
    
    def _parse_switch_ast(
        self,
        lines: List[str],
        start: int,
        target: str
    ) -> Tuple[ASTNode, int]:
        """Parse switch into AST."""
        line = lines[start].strip()
        
        # Extract expression
        match = re.search(r'switch\s*\((.+?)\)', line)
        expr = match.group(1) if match else ""
        
        node = ASTNode(
            node_type="switch",
            label="switch",
            value=expr,
            source_target=target,
            line_number=start
        )
        
        # Find content
        depth = 0
        end = start
        
        for i in range(start, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth == 0 and i > start:
                end = i
                break
        
        # Parse cases
        current_case = None
        for i in range(start + 1, end + 1):
            case_line = lines[i].strip()
            
            if case_line.startswith('case ') or case_line.startswith('default'):
                match = re.match(r'case\s+(\w+(?:::\w+)?)\s*:', case_line)
                if match:
                    case_value = match.group(1)
                    # Abstract target-specific values
                    abstract_value = self._abstract_value(case_value, target)
                    
                    case_node = ASTNode(
                        node_type="case",
                        label=abstract_value,
                        value=case_value,
                        source_target=target,
                        line_number=i
                    )
                    node.add_child(case_node)
                    current_case = case_node
                    
                    # Check for inline return
                    return_match = re.search(r'return\s+(.+?);', case_line)
                    if return_match:
                        ret_value = return_match.group(1)
                        ret_node = ASTNode(
                            node_type="return",
                            label="return",
                            value=ret_value,
                            source_target=target
                        )
                        case_node.add_child(ret_node)
                
                elif case_line.startswith('default'):
                    case_node = ASTNode(
                        node_type="case",
                        label="default",
                        value="default",
                        source_target=target,
                        line_number=i
                    )
                    node.add_child(case_node)
                    current_case = case_node
            
            elif current_case and case_line.startswith('return'):
                ret_value = self._extract_return_value(case_line)
                ret_node = ASTNode(
                    node_type="return",
                    label="return",
                    value=ret_value,
                    source_target=target
                )
                current_case.add_child(ret_node)
        
        return node, end - start + 1
    
    def _parse_if_ast(
        self,
        lines: List[str],
        start: int,
        target: str
    ) -> Tuple[ASTNode, int]:
        """Parse if statement into AST."""
        line = lines[start].strip()
        
        match = re.search(r'if\s*\((.+?)\)', line)
        condition = match.group(1) if match else ""
        
        node = ASTNode(
            node_type="if",
            label="if",
            value=condition,
            source_target=target,
            line_number=start
        )
        
        # Find end
        depth = 0
        end = start
        for i in range(start, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth == 0 and i > start:
                end = i
                break
        
        return node, end - start + 1
    
    def _parse_loop_ast(
        self,
        lines: List[str],
        start: int,
        target: str
    ) -> Tuple[ASTNode, int]:
        """Parse loop into AST."""
        line = lines[start].strip()
        loop_type = "for" if line.startswith("for") else "while"
        
        node = ASTNode(
            node_type="loop",
            label=loop_type,
            value=line,
            source_target=target,
            line_number=start
        )
        
        # Find end
        depth = 0
        end = start
        for i in range(start, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth == 0 and i > start:
                end = i
                break
        
        return node, end - start + 1
    
    def _extract_return_value(self, line: str) -> str:
        """Extract return value from statement."""
        match = re.search(r'return\s+(.+?)\s*;', line)
        return match.group(1) if match else ""
    
    def _extract_func_name(self, line: str) -> str:
        """Extract function name from call."""
        match = re.search(r'(\w+)\s*\(', line)
        return match.group(1) if match else "call"
    
    def _abstract_value(self, value: str, target: str) -> str:
        """Abstract target-specific values to common form."""
        # Replace target names
        abstract = re.sub(
            r'\b(ARM|MIPS|RISCV|X86|RI5CY|xCORE|AArch64)\b',
            'TARGET',
            value
        )
        
        # Abstract relocation types
        abstract = re.sub(r'R_(ARM|MIPS|RISCV|X86)_', 'R_TARGET_', abstract)
        
        return abstract
    
    def align_two(self, ast1: ASTNode, ast2: ASTNode) -> List[AlignmentMapping]:
        """
        Align two ASTs using GumTree-style algorithm.
        
        Phase 1: Top-down matching (greedy, height-based)
        Phase 2: Bottom-up matching (recovery)
        
        Returns:
            List of node mappings
        """
        mappings = []
        
        # Phase 1: Top-down greedy matching
        top_down = self._top_down_match(ast1, ast2)
        mappings.extend(top_down)
        
        # Phase 2: Bottom-up recovery
        matched1 = {m.source for m in mappings}
        matched2 = {m.target for m in mappings}
        
        bottom_up = self._bottom_up_match(ast1, ast2, matched1, matched2)
        mappings.extend(bottom_up)
        
        return mappings
    
    def _top_down_match(
        self,
        ast1: ASTNode,
        ast2: ASTNode
    ) -> List[AlignmentMapping]:
        """Top-down greedy matching phase."""
        mappings = []
        
        # Get all nodes by height (descending)
        nodes1 = self._get_nodes_by_height(ast1)
        nodes2 = self._get_nodes_by_height(ast2)
        
        matched1: Set[ASTNode] = set()
        matched2: Set[ASTNode] = set()
        
        for n1 in nodes1:
            if n1 in matched1:
                continue
            
            if n1.height < self.MIN_HEIGHT:
                continue
            
            best_match = None
            best_sim = 0.0
            
            for n2 in nodes2:
                if n2 in matched2:
                    continue
                
                if n1.node_type != n2.node_type:
                    continue
                
                sim = self._dice_coefficient(n1, n2)
                if sim >= self.MIN_DICE and sim > best_sim:
                    best_match = n2
                    best_sim = sim
            
            if best_match:
                align_type = self._determine_alignment_type(n1, best_match)
                mappings.append(AlignmentMapping(
                    source=n1,
                    target=best_match,
                    alignment_type=align_type,
                    similarity=best_sim
                ))
                
                # Mark all descendants as matched
                matched1.add(n1)
                matched2.add(best_match)
                for d in n1.descendants():
                    matched1.add(d)
                for d in best_match.descendants():
                    matched2.add(d)
        
        return mappings
    
    def _bottom_up_match(
        self,
        ast1: ASTNode,
        ast2: ASTNode,
        matched1: Set[ASTNode],
        matched2: Set[ASTNode]
    ) -> List[AlignmentMapping]:
        """Bottom-up recovery phase."""
        mappings = []
        
        unmatched1 = [n for n in [ast1] + ast1.descendants() if n not in matched1]
        unmatched2 = [n for n in [ast2] + ast2.descendants() if n not in matched2]
        
        for n1 in unmatched1:
            for n2 in unmatched2:
                if n2 in matched2:
                    continue
                
                if n1.node_type == n2.node_type and n1.label == n2.label:
                    align_type = self._determine_alignment_type(n1, n2)
                    mappings.append(AlignmentMapping(
                        source=n1,
                        target=n2,
                        alignment_type=align_type,
                        similarity=self._dice_coefficient(n1, n2)
                    ))
                    matched2.add(n2)
                    break
        
        return mappings
    
    def _get_nodes_by_height(self, ast: ASTNode) -> List[ASTNode]:
        """Get all nodes sorted by height (descending)."""
        all_nodes = [ast] + ast.descendants()
        return sorted(all_nodes, key=lambda n: n.height, reverse=True)
    
    def _dice_coefficient(self, n1: ASTNode, n2: ASTNode) -> float:
        """Calculate Dice coefficient for two subtrees."""
        labels1 = self._get_descendant_labels(n1)
        labels2 = self._get_descendant_labels(n2)
        
        common = labels1 & labels2
        
        if len(labels1) + len(labels2) == 0:
            return 0.0
        
        return 2.0 * len(common) / (len(labels1) + len(labels2))
    
    def _get_descendant_labels(self, node: ASTNode) -> Set[str]:
        """Get set of (type, label) pairs for node and descendants."""
        labels = {(node.node_type, node.label)}
        for d in node.descendants():
            labels.add((d.node_type, d.label))
        return labels
    
    def _determine_alignment_type(
        self,
        n1: ASTNode,
        n2: ASTNode
    ) -> AlignmentType:
        """Determine the type of alignment between two nodes."""
        # Exact match
        if n1.value == n2.value:
            return AlignmentType.IDENTICAL
        
        # Same structure, different target-specific values
        abstract1 = self._abstract_value(n1.value, n1.source_target)
        abstract2 = self._abstract_value(n2.value, n2.source_target)
        
        if abstract1 == abstract2:
            return AlignmentType.EQUIVALENT
        
        # Different
        if n1.source_target != n2.source_target:
            return AlignmentType.TARGET_SPECIFIC
        
        return AlignmentType.MODIFIED
    
    def align_multiple(
        self,
        codes: List[Tuple[str, str]]  # [(target, code), ...]
    ) -> MultiAlignmentResult:
        """
        Align multiple implementations.
        
        Args:
            codes: List of (target_name, source_code) tuples
        
        Returns:
            MultiAlignmentResult with all alignments
        """
        # Parse all to ASTs
        asts = []
        targets = []
        
        for target, code in codes:
            ast = self.parse_to_ast(code, target)
            asts.append(ast)
            targets.append(target)
        
        # Pairwise alignments
        n = len(asts)
        all_alignments = []
        similarity_matrix = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                alignments = self.align_two(asts[i], asts[j])
                all_alignments.append(alignments)
                
                # Calculate similarity
                avg_sim = sum(a.similarity for a in alignments) / max(len(alignments), 1)
                similarity_matrix[i][j] = avg_sim
                similarity_matrix[j][i] = avg_sim
        
        # Set diagonal
        for i in range(n):
            similarity_matrix[i][i] = 1.0
        
        # Find common nodes (present in all with equivalent/identical alignment)
        common_nodes = self._find_common_nodes(asts, all_alignments)
        
        # Find target-specific nodes
        target_specific = self._find_target_specific(asts, common_nodes)
        
        return MultiAlignmentResult(
            asts=asts,
            source_targets=targets,
            alignments=all_alignments,
            common_nodes=common_nodes,
            target_specific_nodes=target_specific,
            similarity_matrix=similarity_matrix
        )
    
    def _find_common_nodes(
        self,
        asts: List[ASTNode],
        alignments: List[List[AlignmentMapping]]
    ) -> List[ASTNode]:
        """Find nodes that are common across all implementations."""
        if not asts:
            return []
        
        # Start with nodes from first AST
        common = set([asts[0]] + asts[0].descendants())
        
        # For each alignment, keep only mapped nodes
        for mapping_list in alignments:
            mapped_sources = {m.source for m in mapping_list 
                           if m.alignment_type in (AlignmentType.IDENTICAL, AlignmentType.EQUIVALENT)}
            common &= mapped_sources
        
        return list(common)
    
    def _find_target_specific(
        self,
        asts: List[ASTNode],
        common_nodes: List[ASTNode]
    ) -> Dict[str, List[ASTNode]]:
        """Find nodes specific to each target."""
        common_set = set(common_nodes)
        target_specific: Dict[str, List[ASTNode]] = {}
        
        for ast in asts:
            target = ast.source_target
            specific = []
            
            for node in [ast] + ast.descendants():
                if node not in common_set:
                    specific.append(node)
            
            target_specific[target] = specific
        
        return target_specific


def align_references(
    references: List[Tuple[str, str]],
    verbose: bool = False
) -> MultiAlignmentResult:
    """
    Convenience function to align multiple reference implementations.
    
    Args:
        references: List of (target_name, code) tuples
        verbose: Enable verbose output
    
    Returns:
        MultiAlignmentResult
    """
    aligner = ASTAligner(verbose=verbose)
    return aligner.align_multiple(references)
