#!/usr/bin/env python3
"""
Quick setup script to create sample company knowledge bases
Run this to create sample IKIO, METCO, and SUNSPRINT knowledge bases
"""

import os
import json

COMPANY_DB_DIR = "company_db"

# Sample company data
COMPANIES_DATA = {
    "IKIO": {
        "texts": [
            "IKIO Energy is a leading solar energy company specializing in large-scale photovoltaic installations across the United States.",
            "We are licensed and registered to operate in all 50 states with active contractor licenses in 48 states.",
            "Our company has completed over 500 MW of solar installations with a perfect safety record - zero lost-time incidents in the past 5 years.",
            "IKIO Energy holds ISO 9001 quality certification and ISO 14001 environmental management certification.",
            "We maintain comprehensive general liability insurance of $10 million and professional liability insurance of $5 million.",
            "Our bonding capacity is up to $50 million per project through our surety partners.",
            "We have experience with Buy American Act (BAA) and Build America Buy America (BABA) compliance on federal projects.",
            "IKIO Energy is certified as a Minority Business Enterprise (MBE) and Small Business Enterprise (SBE) in multiple states.",
            "Our team includes 50+ full-time engineers (electrical, structural, and civil) and 200+ installation technicians.",
            "We have strategic partnerships with Tier 1 solar module manufacturers including JinkoSolar, LONGi, and Canadian Solar.",
            "Our equipment fleet includes 25 specialized solar racking installation vehicles and 10 mobile testing units.",
            "IKIO has completed federal projects for the Department of Defense, Department of Energy, and General Services Administration.",
            "We maintain OSHA 30-hour certification for all supervisors and OSHA 10-hour for all field personnel.",
            "Our warranty program includes 25-year performance warranties on modules and 10-year workmanship warranties.",
            "We provide comprehensive O&M services including remote monitoring, preventive maintenance, and emergency response.",
            "IKIO Energy has financial backing from tier-1 institutional investors with strong balance sheet and credit rating.",
            "Our typical project timeline is 6-12 months from notice to proceed to final commissioning depending on size.",
            "We have experience with Davis-Bacon prevailing wage requirements and union labor agreements.",
            "Our project management team uses industry-standard tools including Primavera P6, Procore, and custom dashboards.",
            "IKIO Energy follows strict environmental protocols including erosion control, waste management, and habitat protection.",
        ]
    },
    "METCO": {
        "texts": [
            "METCO Energy Solutions specializes in renewable energy project development with focus on solar and energy storage systems.",
            "We are licensed contractors in 35 states with particular strength in the Southeast and Midwest regions.",
            "METCO has successfully delivered 350+ MW of solar projects ranging from 5 MW to 50 MW utility-scale installations.",
            "Our company maintains ISO 9001:2015 quality management and ISO 45001 occupational health and safety certifications.",
            "We carry $15 million general liability insurance and $7 million professional liability coverage.",
            "Our bonding capacity extends to $75 million for single projects through AA-rated surety companies.",
            "METCO has extensive experience with NEPA environmental reviews and state-level permitting processes.",
            "We are certified as Small Business Administration (SBA) 8(a) participant and WOSB (Women-Owned Small Business).",
            "Our engineering team consists of 35 PE-licensed engineers specializing in electrical power systems and civil engineering.",
            "We maintain direct relationships with inverter manufacturers including SMA, Power Electronics, and Sungrow.",
            "Our equipment inventory includes specialized construction machinery, testing equipment, and safety gear for 150+ workers.",
            "METCO has clearance to work on Department of Defense installations and maintains NISPOM compliance for classified sites.",
            "All project managers hold PMP certification and our safety team includes 5 Certified Safety Professionals (CSP).",
            "We offer extended warranties up to 30 years on solar installations and 15-year warranties on electrical balance of system.",
            "METCO provides 24/7 operations and maintenance support through our dedicated O&M division with regional service centers.",
            "Our financial position is strong with annual revenue of $200M+ and access to project financing through major banks.",
            "Typical project execution timeline is 8-14 months including permitting, procurement, construction, and commissioning phases.",
            "We have experience with union labor requirements and maintain labor agreements with IBEW and other trade unions.",
            "METCO uses integrated project management platform combining scheduling, cost control, quality assurance, and document management.",
            "We follow comprehensive environmental management plans including stormwater management, dust control, and wildlife protection.",
        ]
    },
    "SUNSPRINT": {
        "texts": [
            "SUNSPRINT Renewables is a full-service solar EPC contractor with nationwide capabilities and 15 years of industry experience.",
            "We hold active contractor licenses in 42 states and maintain A+ rating with Better Business Bureau.",
            "Our portfolio includes 400+ MW of completed solar projects with 98% on-time, on-budget delivery record.",
            "SUNSPRINT maintains ISO 9001, ISO 14001, and ISO 45001 triple certification demonstrating quality, environmental, and safety excellence.",
            "We carry comprehensive insurance including $20 million general liability, $10 million professional liability, and $25 million umbrella coverage.",
            "Our bonding capacity is $100 million aggregate with individual project capacity up to $60 million.",
            "SUNSPRINT has successfully navigated Buy American requirements on 50+ federal projects with documented compliance.",
            "We are certified Small Business, SDVOSB (Service-Disabled Veteran-Owned), and qualify for various state diversity goals.",
            "Our technical team includes 75 engineers (PE-licensed), 25 project managers (PMP-certified), and 300+ skilled craftworkers.",
            "We have master agreements with all major equipment suppliers including modules, inverters, trackers, and balance of system components.",
            "SUNSPRINT owns specialized construction equipment valued at $15M including cranes, pile drivers, trenchers, and testing equipment.",
            "We maintain SECRET facility clearance and have completed classified projects for DOD, DOE, and intelligence agencies.",
            "Our safety program exceeds OSHA requirements with EMR of 0.65 and zero fatalities in company history.",
            "We provide industry-leading warranties: 30-year performance warranty, 20-year workmanship warranty, and 15-year balance of system warranty.",
            "SUNSPRINT O&M services include SCADA monitoring, predictive maintenance using AI analytics, and guaranteed uptime of 98%+.",
            "Financial strength is demonstrated by $500M+ annual revenue, investment-grade credit rating, and partnerships with major financial institutions.",
            "Our accelerated delivery capability can complete projects 20% faster than industry average while maintaining quality standards.",
            "We have extensive experience with prevailing wage compliance, certified payroll, and collective bargaining agreements with 10+ trade unions.",
            "SUNSPRINT utilizes cutting-edge project controls including real-time dashboards, drone progress monitoring, and AI-powered scheduling.",
            "Our environmental compliance record is perfect with zero violations and proactive sustainability initiatives including carbon-neutral operations.",
        ]
    }
}

def create_company_knowledge_bases():
    """Create sample company knowledge base JSON files"""
    
    # Create company_db directory if it doesn't exist
    os.makedirs(COMPANY_DB_DIR, exist_ok=True)
    
    for company_name, data in COMPANIES_DATA.items():
        file_path = os.path.join(COMPANY_DB_DIR, f"{company_name}.json")
        
        # Write JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"[OK] Created {file_path} with {len(data['texts'])} knowledge chunks")
    
    print(f"\n[SUCCESS] Created knowledge bases for {len(COMPANIES_DATA)} companies!")
    print(f"[INFO] Files are in the '{COMPANY_DB_DIR}' directory")
    print("\n[NEXT STEPS] You can now:")
    print("   1. Edit these JSON files to add more company-specific information")
    print("   2. Or upload company context files through the web interface")
    print("   3. Or replace these with your actual company profiles")

if __name__ == "__main__":
    create_company_knowledge_bases()

