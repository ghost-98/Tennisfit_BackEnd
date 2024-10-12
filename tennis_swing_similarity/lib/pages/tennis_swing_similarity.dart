import 'package:flutter/material.dart';
import 'my_page.dart';
import 'swing_similarity.dart';
import 'sns.dart';
import 'tennis_map.dart';

class TennisSwingSimilarity extends StatefulWidget {
  const TennisSwingSimilarity({super.key});

  @override
  State<TennisSwingSimilarity> createState() => _TennisSwingSimilarityState();
}

class _TennisSwingSimilarityState extends State<TennisSwingSimilarity> {
  int _bottomItemIndex = 0;

  final List<Widget> _pages = [
    const SwingSimilarity(),
    const TennisMap(),
    const SNS(),
    const MyPage()
  ];

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
        title: 'Tennis Swing Similarity',
        home: Scaffold(
          appBar: AppBar(
            backgroundColor: const Color.fromRGBO(255, 255, 255, 0),
            elevation: 0,
            centerTitle: false,
            titleTextStyle: const TextStyle(
                color: Colors.black,
                fontSize: 30,
                fontWeight: FontWeight.w600
            ),
            title: const Text("Tennis Swing Similarity"),),
          body: _pages[_bottomItemIndex],
          bottomNavigationBar: Container(
            height: 110,
            decoration: BoxDecoration(
                color: Colors.white,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.2),
                    blurRadius: 5.0,
                    spreadRadius: 0.0,
                    offset: const Offset(0,-1),
                  )
                ]
            ),
            child: BottomNavigationBar(
              items: const[
                BottomNavigationBarItem(
                  icon: Icon(Icons.accessibility),
                  label: '유사도 검사',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.map_outlined),
                  label: '테니스 지도',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.textsms_outlined),
                  label: '소통',
                ),
                BottomNavigationBarItem(
                  icon: Icon(Icons.account_circle),
                  label: '내 정보',
                ),
              ],
              currentIndex: _bottomItemIndex,
              selectedFontSize: 16,
              selectedItemColor: Colors.blue,
              unselectedItemColor: Colors.black,
              onTap: (int index) {
                setState(() {
                  _bottomItemIndex = index;
                });
              },
            ),
          ),
        )
    );
  }
}