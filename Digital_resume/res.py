import streamlit as st

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Resume", "Projects", "Contact"])

# Resume Section
if page == "Resume":
    st.title("Yashvant Gupta")
    st.subheader("Data Analytics")

    st.write("📍 Satna, Madhya Pradesh")
    st.write("📧 guptashanu227@gmail.com")
    st.write("📞 9301264010")
    st.write("[LinkedIn](https://www.linkedin.com/in/yashvant-gupta-2a08b5380/)")

    st.markdown("### Profile")
    st.write("""
    Data Analytics student with hands-on experience in building AI-powered dashboards 
    and analyzing business data. Proficient in Python, SQL, and Power BI for data cleaning, 
    analysis, and visualization. Skilled in extracting insights from data, creating interactive 
    dashboards, and supporting data-driven decision-making.
    """)

    st.markdown("### Skills")
    st.write("- SQL Queries\n- Data Cleaning\n- Problem Solving\n- Data Visualization\n- Dashboard Building\n- Effective Communication\n- Critical Thinking\n- Data Analysis")

    st.markdown("### Technical Skills")
    st.write("- Python\n- Power BI\n- Excel\n- Pandas\n- MySQL\n- PostgreSQL\n- Numpy\n- Matplotlib\n- Jupyter Notebook")

    st.markdown("### Certificates")
    st.write("- Data Science Certification\n- Python Programming\n- Power BI\n- SQL, Excel")

    st.markdown("### Education")
    st.write("**B.Tech Computer Science Engineering** - VITS College, Satna (RGPV Board) | GPA: 8.00+ | 2023–2027")
    st.write("**High School** - G,C,R,D,S High School Kitaha (MP) | Percentage: 95% | 2020–2021")

# Projects Section
elif page == "Projects":
    st.title("Projects")

    st.subheader("AI-Powered Business Analytics Dashboard")
    st.write("""
    Developed an interactive business analytics dashboard using Streamlit and Pandas to analyze datasets 
    and generate insights. Implemented KPIs with trend analysis and integrated AI features for automated insights.
    **Technologies:** Python, Streamlit, Pandas, NumPy, Matplotlib
    """)

    st.subheader("Sales Data Analysis Dashboard")
    st.write("""
    Analyzed 10,000+ rows of sales data using Excel and SQL to identify key business insights. 
    Built an interactive Power BI dashboard to visualize KPIs such as revenue, profit, and growth trends.
    **Tools:** Excel, SQL, Power BI, Python
    """)

# Contact Section
elif page == "Contact":
    st.title("Contact Me")
    st.write("📧 guptashanu227@gmail.com")
    st.write("📞 9301264010")
    st.write("[LinkedIn](https://www.linkedin.com/in/yashvant-gupta-2a08b5380/)")
    st.write("📍 Satna, Madhya Pradesh")
