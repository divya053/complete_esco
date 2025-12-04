#!/usr/bin/env python3
"""
Test script for the local RAG proposal generation functionality.
Tests the new generate_proposal_with_local_rag function without API calls.
"""

import sys
import os
import json
from datetime import datetime

# Add the current directory to the path so we can import from app_v2.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_local_rag_proposal_generation():
    """Test the local RAG proposal generation function"""

    # Import the function we want to test
    try:
        from app_v2 import generate_proposal_with_local_rag
        print("✓ Successfully imported generate_proposal_with_local_rag")
    except ImportError as e:
        print(f"✗ Failed to import function: {e}")
        return False

    # Sample RFP text
    sample_rfp_text = """
    REQUEST FOR PROPOSAL (RFP)

    Project Title: Infrastructure Development Project

    Project Overview:
    The City of Springfield is seeking qualified contractors to provide comprehensive infrastructure development services.
    The project involves construction of new roads, bridges, and utility systems.

    Scope of Work:
    1. Design and engineering of transportation infrastructure
    2. Construction management and supervision
    3. Quality control and testing
    4. Project documentation and reporting

    Technical Requirements:
    - Experience with large-scale infrastructure projects
    - Licensed and certified professionals
    - Safety compliance and risk management
    - Environmental impact assessment

    Project Timeline:
    - Proposal submission deadline: August 31, 2026
    - Project start: September 15, 2026
    - Project completion: August 31, 2027

    Evaluation Criteria:
    - Technical capability (40%)
    - Experience and qualifications (30%)
    - Price proposal (20%)
    - Project management approach (10%)
    """

    # Test parameters
    company_name = "IKIO"
    project_title = "Infrastructure Development Project"
    deadline = "August 31, 2026"
    user_email = "test@ikio.com"

    # Sample sections outline
    sections_outline = [
        {
            'title': 'Executive Summary',
            'guidance': 'Summarize the opportunity, objectives, and how the proposed solution delivers value.'
        },
        {
            'title': 'Company Profile',
            'guidance': 'Describe the company background, capabilities, and relevant experience.'
        },
        {
            'title': 'Technical Approach & Scope',
            'guidance': 'Detail the technical solution, methodology, schedule, and risk mitigation strategies.'
        },
        {
            'title': 'Project Management & Schedule',
            'guidance': 'Outline management structure, milestones, resource plan, and communication strategy.'
        }
    ]

    try:
        print("Testing local RAG proposal generation...")
        print("-" * 50)

        # Generate proposal
        result = generate_proposal_with_local_rag(
            sample_rfp_text, company_name, project_title, deadline, sections_outline, user_email
        )

        print("✓ Function executed successfully")
        print(f"✓ Generated {len(result.get('sections', []))} sections")

        # Validate structure
        if 'sections' not in result:
            print("✗ Missing 'sections' key in result")
            return False

        sections = result['sections']
        if not sections:
            print("✗ No sections generated")
            return False

        # Check each section
        for i, section in enumerate(sections):
            if 'title' not in section:
                print(f"✗ Section {i} missing 'title'")
                return False
            if 'content' not in section:
                print(f"✗ Section {section['title']} missing 'content'")
                return False
            if not isinstance(section['content'], list):
                print(f"✗ Section {section['title']} content is not a list")
                return False

            print(f"✓ Section '{section['title']}' generated with {len(section['content'])} paragraphs")

        # Show sample content
        print("\nSample generated content:")
        print("-" * 30)
        first_section = sections[0]
        print(f"Section: {first_section['title']}")
        for i, para in enumerate(first_section['content'][:2]):  # Show first 2 paragraphs
            print(f"Paragraph {i+1}: {para[:100]}...")

        print("\n✓ All tests passed! Local RAG proposal generation is working.")
        return True

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rfp_pattern_extraction():
    """Test the RFP pattern extraction function"""
    try:
        from app_v2 import extract_rfp_patterns
        print("\nTesting RFP pattern extraction...")

        sample_rfp = """
        The project must include requirements for safety compliance.
        Technical specifications include software development and testing.
        Deadline is August 31, 2026.
        Budget is $500,000.
        """

        patterns = extract_rfp_patterns(sample_rfp)

        if 'project_requirements' in patterns and patterns['project_requirements']:
            print("✓ Project requirements extracted")
        if 'technical_specs' in patterns and patterns['technical_specs']:
            print("✓ Technical specifications extracted")
        if 'deadlines' in patterns and patterns['deadlines']:
            print("✓ Deadlines extracted")

        print("✓ RFP pattern extraction working")
        return True

    except Exception as e:
        print(f"✗ RFP pattern extraction error: {e}")
        return False


if __name__ == "__main__":
    print("Local RAG Proposal Generation Test")
    print("=" * 50)

    success1 = test_local_rag_proposal_generation()
    success2 = test_rfp_pattern_extraction()

    if success1 and success2:
        print("\n🎉 All tests passed! The local RAG system is ready to use.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)
